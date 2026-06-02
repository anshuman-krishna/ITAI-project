from typing import Generic, TypeVar
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    # thin data-access layer. services depend on repositories, not on the orm
    # directly, so storage details stay isolated from business logic.

    def __init__(self, model: type[ModelType], session: AsyncSession) -> None:
        self.model = model
        self.session = session

    async def get(self, id: UUID) -> ModelType | None:
        return await self.session.get(self.model, id)

    async def list(self, limit: int = 100, offset: int = 0) -> list[ModelType]:
        result = await self.session.execute(
            select(self.model).limit(limit).offset(offset)
        )
        return list(result.scalars().all())

    async def add(self, entity: ModelType) -> ModelType:
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def delete(self, entity: ModelType) -> None:
        await self.session.delete(entity)
