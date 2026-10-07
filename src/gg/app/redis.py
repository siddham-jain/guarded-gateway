from redis.asyncio import BlockingConnectionPool, Redis


def connect_redis(url: str, *, max_connections: int, pool_timeout_s: float) -> Redis:
    # the default pool raises once 100 connections are busy, which fails budgets open under load;
    # a blocking pool queues for a free connection instead and only errors after pool_timeout_s
    pool = BlockingConnectionPool.from_url(url, max_connections=max_connections, timeout=pool_timeout_s)
    return Redis.from_pool(pool)
