# docker

The development stack is defined by the root `docker-compose.yml`. Each app keeps its
own `Dockerfile` next to its code; their build context is the repo root so they can pull
in shared packages (`packages/ml-core`, `packages/shared`).

This folder holds Docker concerns that aren't tied to a single app — production compose
overrides, base images, or volume tooling — as they're added.
