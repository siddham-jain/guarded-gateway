"""redis lua scripts; each touches one hash-tagged scope ({k:<key_id>}) so it stays single-slot.

the now override argument exists only for virtual-time tests; production callers always pass ''.
"""

_NOW = """
local function clock(override)
  if override ~= '' then return tonumber(override) end
  local t = redis.call('TIME')
  return tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
end
"""

# KEYS: 1 rpm hash, 2 tpm hash, 3 concurrency zset
# ARGV: 1 rpm, 2 rpm_cap, 3 tpm, 4 tpm_cap, 5 need, 6 conc_max, 7 lease_id, 8 lease_ttl_ms,
#       9 bucket_ttl_ms, 10 now override
# reply: allowed, reason, rpm level, tpm level, retry ms, in use, rpm reset ms, tpm reset ms, now ms
ADMIT = (
    _NOW
    + """
local now = clock(ARGV[10])
local rpm, rpm_cap, tpm, tpm_cap = tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3]), tonumber(ARGV[4])
local need, conc_max, lease, ttl = tonumber(ARGV[5]), tonumber(ARGV[6]), ARGV[7], tonumber(ARGV[8])
local bucket_ttl = tonumber(ARGV[9])

local function level(key, rate, cap)
  if rate <= 0 then return nil end
  local s = redis.call('HMGET', key, 'v', 'ts')
  local v = tonumber(s[1]) or cap
  local ts = tonumber(s[2]) or now
  return math.min(cap, v + math.max(0, now - ts) * rate / 60000.0)
end
local function wait_ms(v, want, rate)
  if v == nil or v >= want then return 0 end
  return math.ceil((want - v) * 60000.0 / rate)
end
local function reset_ms(v, rate, cap)
  if v == nil then return 0 end
  return math.max(0, math.ceil((cap - v) * 60000.0 / rate))
end
local function floor(v)
  if v == nil then return 0 end
  return math.floor(v)
end

local in_use = 0
if conc_max > 0 then
  redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', now)
  in_use = redis.call('ZCARD', KEYS[3])
end
local rv, tv = level(KEYS[1], rpm, rpm_cap), level(KEYS[2], tpm, tpm_cap)

local function reply(allowed, reason, retry, used)
  return {allowed, reason, floor(rv), floor(tv), retry, used,
          reset_ms(rv, rpm, rpm_cap), reset_ms(tv, tpm, tpm_cap), now}
end

if tv ~= nil and need > tpm_cap then return reply(0, 'tokens_exceed_limit', 0, in_use) end
if conc_max > 0 and in_use >= conc_max then return reply(0, 'concurrency', 1000, in_use) end
local rw, tw = wait_ms(rv, 1, rpm), wait_ms(tv, need, tpm)
if rw > 0 or tw > 0 then
  local reason = 'tpm'
  if rw >= tw then reason = 'rpm' end
  return reply(0, reason, math.max(rw, tw), in_use)
end

if rv ~= nil then
  rv = rv - 1
  redis.call('HSET', KEYS[1], 'v', rv, 'ts', now)
  redis.call('PEXPIRE', KEYS[1], bucket_ttl)
end
if tv ~= nil then
  tv = tv - need
  redis.call('HSET', KEYS[2], 'v', tv, 'ts', now)
  redis.call('PEXPIRE', KEYS[2], bucket_ttl)
end
if conc_max > 0 then
  redis.call('ZADD', KEYS[3], now + ttl, lease)
  redis.call('PEXPIRE', KEYS[3], ttl + 60000)
  in_use = in_use + 1
end
return reply(1, 'ok', 0, in_use)
"""
)

# KEYS: 1 tpm hash, 2 concurrency zset
# ARGV: 1 tpm, 2 tpm_cap, 3 delta (actual - reserved), 4 lease_id, 5 bucket_ttl_ms, 6 now override
FINISH = (
    _NOW
    + """
local tpm, cap, delta = tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3])
if tpm > 0 and delta ~= 0 then
  local now = clock(ARGV[6])
  local s = redis.call('HMGET', KEYS[1], 'v', 'ts')
  local v = tonumber(s[1]) or cap
  local ts = tonumber(s[2]) or now
  v = math.min(cap, v + math.max(0, now - ts) * tpm / 60000.0)
  v = math.max(-cap, math.min(cap, v - delta))
  redis.call('HSET', KEYS[1], 'v', v, 'ts', now)
  redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[5]))
end
redis.call('ZREM', KEYS[2], ARGV[4])
return 1
"""
)

_HOLD = """
local function parse(raw)
  local amount, state, pkeys = string.match(raw, '^(%-?%d+)|(%a+)|(.*)$')
  local list = {}
  for pk in string.gmatch(pkeys, '[^,]+') do list[#list + 1] = pk end
  return tonumber(amount), state, pkeys, list
end
"""

# KEYS: 1..P period hashes, P+1 holds zset, P+2 the new hold's string key
# ARGV: 1 P, 2..P+1 caps (-1 = uncapped), P+2 amount, P+3 hold_id, P+4 hold_ttl_ms, P+5 sweep_limit,
#       P+6 hold key prefix, P+7 tombstone_ms, P+8 now override
# reply: ok, swept count, refused period index (0 when ok), then each period's spent + reserved before
RESERVE = (
    _NOW
    + _HOLD
    + """
local p = tonumber(ARGV[1])
local amount, hold_id, ttl = tonumber(ARGV[p + 2]), ARGV[p + 3], tonumber(ARGV[p + 4])
local sweep_limit, prefix, tomb = tonumber(ARGV[p + 5]), ARGV[p + 6], tonumber(ARGV[p + 7])
local now = clock(ARGV[p + 8])
local holds = KEYS[p + 1]

-- expired holds (crashed or never settled) count as spent; settle corrects them later
local swept = 0
for _, hid in ipairs(redis.call('ZRANGEBYSCORE', holds, '-inf', now, 'LIMIT', 0, sweep_limit)) do
  local raw = redis.call('GET', prefix .. hid)
  if raw then
    local amt, state, pkeys, list = parse(raw)
    if state == 'open' then
      for _, pk in ipairs(list) do
        redis.call('HINCRBY', pk, 'reserved', -amt)
        redis.call('HINCRBY', pk, 'spent', amt)
      end
      redis.call('SET', prefix .. hid, amt .. '|swept|' .. pkeys, 'PX', tomb)
      swept = swept + 1
    end
  end
  redis.call('ZREM', holds, hid)
end

local used = {}
for i = 1, p do
  local s = redis.call('HMGET', KEYS[i], 'spent', 'reserved')
  used[i] = (tonumber(s[1]) or 0) + (tonumber(s[2]) or 0)
end
for i = 1, p do
  local cap = tonumber(ARGV[1 + i])
  if cap >= 0 and used[i] + amount > cap then
    local out = {0, swept, i}
    for j = 1, p do out[#out + 1] = used[j] end
    return out
  end
end

local pkeys = {}
for i = 1, p do
  redis.call('HINCRBY', KEYS[i], 'reserved', amount)
  pkeys[i] = KEYS[i]
end
redis.call('ZADD', holds, now + ttl, hold_id)
-- open holds carry no ttl so volatile-lru can never drop a reservation
redis.call('SET', KEYS[p + 2], amount .. '|open|' .. table.concat(pkeys, ','))
local out = {1, swept, 0}
for j = 1, p do out[#out + 1] = used[j] end
return out
"""
)

# KEYS: 1 hold string key
# ARGV: 1 new amount, 2.. caps aligned with the hold's period keys (-1 = uncapped)
# reply: 1 applied | 0 refused, then the hold state ('missing' when the hold is gone)
ADJUST = (
    _HOLD
    + """
local raw = redis.call('GET', KEYS[1])
if not raw then return {0, 'missing'} end
local amt, state, pkeys, list = parse(raw)
if state ~= 'open' then return {0, state} end
local new = tonumber(ARGV[1])
local diff = new - amt
if diff > 0 then
  for i, pk in ipairs(list) do
    local cap = tonumber(ARGV[1 + i])
    if cap >= 0 then
      local s = redis.call('HMGET', pk, 'spent', 'reserved')
      if (tonumber(s[1]) or 0) + (tonumber(s[2]) or 0) + diff > cap then return {0, 'open'} end
    end
  end
end
if diff ~= 0 then
  for _, pk in ipairs(list) do redis.call('HINCRBY', pk, 'reserved', diff) end
  redis.call('SET', KEYS[1], new .. '|open|' .. pkeys)
end
return {1, 'open'}
"""
)

# KEYS: 1 hold string key, 2 holds zset, 3.. current period keys (charged only when the hold is missing)
# ARGV: 1 actual, 2 hold_id, 3 tombstone_ms
# reply: the state the hold was in (open | swept | settled | missing)
SETTLE = (
    _HOLD
    + """
local actual = tonumber(ARGV[1])
local raw = redis.call('GET', KEYS[1])
if not raw then
  for i = 3, #KEYS do redis.call('HINCRBY', KEYS[i], 'spent', actual) end
  return 'missing'
end
local amt, state, pkeys, list = parse(raw)
if state == 'open' then
  for _, pk in ipairs(list) do
    redis.call('HINCRBY', pk, 'reserved', -amt)
    redis.call('HINCRBY', pk, 'spent', actual)
  end
  redis.call('ZREM', KEYS[2], ARGV[2])
elseif state == 'swept' then
  for _, pk in ipairs(list) do redis.call('HINCRBY', pk, 'spent', actual - amt) end
else
  return state
end
redis.call('SET', KEYS[1], actual .. '|settled|' .. pkeys, 'PX', tonumber(ARGV[3]))
return state
"""
)
