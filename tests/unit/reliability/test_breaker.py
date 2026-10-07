import pytest

from gg.core.clock import FakeClock
from gg.reliability.breaker import BreakerConfig, BreakerRegistry, CircuitBreaker, Cooldown
from tests.unit.reliability.fakes import TransitionLog, deployment


def breaker(clock: FakeClock, log: TransitionLog | None = None) -> CircuitBreaker:
    return CircuitBreaker("a/m", "a", BreakerConfig(), clock, log)


def fail_once(b: CircuitBreaker) -> None:
    permit = b.try_acquire()
    assert permit is not None
    permit.failure()


def succeed_once(b: CircuitBreaker) -> None:
    permit = b.try_acquire()
    assert permit is not None
    permit.success()


def test_opens_after_three_consecutive_failures(clock: FakeClock) -> None:
    b = breaker(clock)
    fail_once(b)
    fail_once(b)
    assert b.state == "closed"
    fail_once(b)

    assert b.state == "open"
    assert not b.available()
    assert b.try_acquire() is None
    assert b.retry_in() == pytest.approx(30)


def test_success_resets_consecutive_count(clock: FakeClock) -> None:
    b = breaker(clock)
    for _ in range(5):
        fail_once(b)
        fail_once(b)
        succeed_once(b)
        clock.advance(61)
    assert b.state == "closed"


def test_opens_on_failure_rate_over_window(clock: FakeClock) -> None:
    b = breaker(clock)
    # f f s f f s f f s f -> 7 of 10 failed, never 3 in a row
    for i in range(10):
        if i % 3 == 2:
            succeed_once(b)
        else:
            fail_once(b)
    assert b.state == "open"


def test_old_calls_leave_the_window(clock: FakeClock) -> None:
    b = breaker(clock)
    for i in range(9):
        (succeed_once if i % 3 == 2 else fail_once)(b)
    clock.advance(61)
    fail_once(b)
    assert b.state == "closed"


def test_half_open_admits_one_probe_then_closes(clock: FakeClock) -> None:
    log = TransitionLog()
    b = breaker(clock, log)
    for _ in range(3):
        fail_once(b)
    clock.advance(30)

    assert b.state == "half_open"
    probe = b.try_acquire()
    assert probe is not None
    assert probe.probe
    assert b.try_acquire() is None
    probe.success()

    assert b.state == "closed"
    assert [(old, new, reason) for _, old, new, reason in log.events] == [
        ("closed", "open", "trip"),
        ("open", "half_open", "cooldown_elapsed"),
        ("half_open", "closed", "probe_ok"),
    ]


def test_failed_probe_doubles_cooldown_up_to_cap(clock: FakeClock) -> None:
    b = breaker(clock)
    for _ in range(3):
        fail_once(b)
    expected = [60, 120, 240, 300, 300]
    for cooldown in expected:
        clock.advance(b.retry_in())
        fail_once(b)
        assert b.retry_in() == pytest.approx(cooldown)

    clock.advance(300)
    succeed_once(b)
    for _ in range(3):
        fail_once(b)
    assert b.retry_in() == pytest.approx(30)


def test_neutral_probe_returns_permit(clock: FakeClock) -> None:
    b = breaker(clock)
    for _ in range(3):
        fail_once(b)
    clock.advance(30)
    probe = b.try_acquire()
    assert probe is not None
    probe.neutral()

    assert b.state == "half_open"
    assert b.try_acquire() is not None


def test_permit_settles_once(clock: FakeClock) -> None:
    permit = breaker(clock).try_acquire()
    assert permit is not None
    permit.success()
    with pytest.raises(RuntimeError):
        permit.failure()


def test_force_open_keeps_later_until(clock: FakeClock) -> None:
    b = breaker(clock)
    b.force_open(Cooldown(600, "auth"))
    b.force_open(Cooldown(30, "rate_limited"))
    assert b.retry_in() == pytest.approx(600)
    assert b.snapshot().reason == "auth"

    b.force_open(Cooldown(3600, "billing"))
    assert b.retry_in() == pytest.approx(3600)


def test_late_failure_counts_in_closed_state(clock: FakeClock) -> None:
    b = breaker(clock)
    for _ in range(3):
        succeed_once(b)
        b.record_late_failure()
    assert b.snapshot().consecutive_failures == 1
    b.record_late_failure()
    b.record_late_failure()
    assert b.state == "open"


def test_registry_provider_scope_reaches_existing_and_new_breakers(clock: FakeClock) -> None:
    registry = BreakerRegistry(clock)
    a1, a2, b1 = deployment("a/one"), deployment("a/two"), deployment("b/one")
    registry.get(a1)

    registry.force_open(a1, Cooldown(600, "auth"), scope="provider")

    assert registry.get(a1).state == "open"
    assert registry.get(a2).state == "open"
    assert registry.get(a2).retry_in() == pytest.approx(600)
    assert registry.get(b1).state == "closed"

    clock.advance(600)
    assert registry.get(deployment("a/three")).state == "closed"


def test_snapshot_reports_state(clock: FakeClock) -> None:
    registry = BreakerRegistry(clock)
    registry.force_open(deployment("a/one"), Cooldown(10, "per_day"))
    snap = registry.snapshot()["a/one"]
    assert (snap.state, snap.reason, snap.open_for_s) == ("open", "per_day", 10)
