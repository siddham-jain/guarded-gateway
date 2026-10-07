import httpx2
import pytest

from gg.config.settings import Settings
from tests.unit.api.fakes import build_app, make_services


@pytest.mark.parametrize(("env", "status"), [("dev", 200), ("test", 200), ("prod", 404)])
async def test_playground_hidden_in_prod(env: str, status: int) -> None:
    settings = Settings(env=env, log_format="json")  # pyright: ignore[reportArgumentType]
    app = build_app(make_services(settings=settings))
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app)) as client:
        res = await client.get("http://gg.test/playground")
    assert res.status_code == status
    if status == 200:
        assert res.headers["cache-control"] == "no-store"
        assert res.headers["x-frame-options"] == "DENY"
