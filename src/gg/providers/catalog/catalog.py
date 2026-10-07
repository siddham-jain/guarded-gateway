from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time

from gg.core.deployment import Deployment, Pricing
from gg.core.schema import ChatRequest
from gg.providers.base import AliasRef, DeploymentRef, PublicModel, Resolution
from gg.providers.catalog.capabilities import AdjustPolicy, CapabilityChecker
from gg.providers.catalog.schema import AliasSpec
from gg.providers.runtime import ProviderRuntime


@dataclass(frozen=True, slots=True)
class Group:
    chain: tuple[str, ...]
    chain_with_tools: tuple[str, ...] | None


class Catalog:
    """implements gg.providers.base.ModelCatalog over the resolved models.yaml (post profile, post env)"""

    def __init__(
        self,
        *,
        deployments: Mapping[str, Deployment],
        groups: Mapping[str, Group],
        aliases: Mapping[str, AliasSpec],
        canonical: frozenset[str],
        public: frozenset[str],
        providers: Mapping[str, ProviderRuntime],
        bare_names: bool,
        checker: CapabilityChecker,
        catalog_hash: str,
    ) -> None:
        self._deployments = dict(deployments)
        self._groups = dict(groups)
        self._aliases = dict(aliases)
        self._canonical = canonical
        self._public = public
        self.providers: Mapping[str, ProviderRuntime] = dict(providers)
        self._checker = checker
        self._hash = catalog_hash
        self._bare: dict[str, str] = {}
        if bare_names:
            counts: dict[str, list[str]] = {}
            for dep in self._deployments.values():
                if dep.enabled and dep.canonical_model is None:
                    counts.setdefault(dep.upstream_model, []).append(dep.id)
            self._bare = {name: ids[0] for name, ids in counts.items() if len(ids) == 1}

    @property
    def deployments(self) -> Sequence[Deployment]:
        return tuple(self._deployments.values())

    @property
    def alias_names(self) -> Sequence[str]:
        return tuple(self._aliases)

    @property
    def group_names(self) -> Sequence[str]:
        return tuple(self._groups)

    @property
    def hash(self) -> str:
        return self._hash

    def resolve(self, model: str, /) -> Resolution | None:
        alias = self._aliases.get(model)
        if alias is not None:
            return AliasRef(model, "router" if alias.router else "group")
        if model in self._canonical:
            return AliasRef(model, "group") if self._enabled(self._groups[model].chain) else None
        dep = self._deployments.get(model)
        if dep is None and model in self._bare and model not in self._groups:
            dep = self._deployments[self._bare[model]]
        if dep is not None and dep.enabled:
            return DeploymentRef(dep)
        return None

    def get(self, deployment_id: str, /) -> Deployment:
        return self._deployments[deployment_id]

    def chain(self, group: str, request: ChatRequest, /) -> list[Deployment]:
        alias = self._aliases.get(group)
        if alias is not None and alias.group is not None:
            group = alias.group
        spec = self._groups.get(group)
        if spec is None:
            return []
        ids = spec.chain_with_tools if request.has_tools() and spec.chain_with_tools else spec.chain
        return self._enabled(ids)

    def groups_of(self, alias: str, /) -> tuple[str, ...]:
        spec = self._aliases.get(alias)
        if spec is None:
            return ()
        return tuple(spec.groups) if spec.router else (spec.group or "",)

    def deployments_for(self, model: str, /) -> list[Deployment]:
        resolution = self.resolve(model)
        if resolution is None:
            return []
        if isinstance(resolution, DeploymentRef):
            return [resolution.deployment]
        names = self.groups_of(model) or (model,)
        ids: list[str] = []
        for name in names:
            spec = self._groups.get(name)
            if spec is not None:
                ids.extend(spec.chain)
                ids.extend(spec.chain_with_tools or ())
        return self._enabled(list(dict.fromkeys(ids)))

    def compatible(
        self, request: ChatRequest, deployments: Sequence[Deployment], policy: AdjustPolicy | None = None
    ) -> list[Deployment]:
        return self._checker.compatible(request, deployments, policy)

    def price_for(self, deployment_id: str, at: datetime) -> Pricing | None:
        return self._deployments[deployment_id].pricing.at(at)

    def list_public(self, allowed: Callable[[str], bool], /) -> Sequence[PublicModel]:
        rows = [
            PublicModel(id=name, owned_by="gg", created=0)
            for name in self._aliases
            if allowed(name) and self.resolve(name) is not None
        ]
        for name in sorted(self._canonical):
            if (
                allowed(name)
                and self.resolve(name) is not None
                and self._groups[name].chain[0] in self._public
            ):
                first = self._deployments[self._groups[name].chain[0]]
                rows.append(
                    PublicModel(
                        id=name,
                        owned_by="gg",
                        created=_created(first),
                        context_window=first.capabilities.context,
                    )
                )
        rows.extend(
            PublicModel(
                id=dep.id,
                owned_by=dep.provider,
                created=_created(dep),
                context_window=dep.capabilities.context,
                shutdown_date=dep.shutdown_date.isoformat() if dep.shutdown_date else None,
            )
            for dep in sorted(self._deployments.values(), key=lambda d: d.id)
            if dep.enabled and dep.id in self._public and allowed(dep.id)
        )
        return rows

    def _enabled(self, ids: Sequence[str]) -> list[Deployment]:
        return [d for d in (self._deployments.get(i) for i in ids) if d is not None and d.enabled]


def _created(dep: Deployment) -> int:
    periods = dep.pricing.periods
    if not periods:
        return 0
    first = min(p.effective_from for p in periods)
    return int(datetime.combine(first, time(), tzinfo=UTC).timestamp())
