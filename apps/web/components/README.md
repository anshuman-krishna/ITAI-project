# components

UI components, grouped by responsibility rather than dumped in one folder. Create these
subfolders as features arrive:

```
layout/       shells, navigation, page frames
charts/       recharts wrappers (metrics, distributions, embeddings)
forms/         inputs and form controls
tables/        data tables
experiments/   experiment creation, management, comparison
datasets/      dataset upload and summaries
models/        model selection and details
ui/            shadcn/ui primitives (button, card, ...) live here
```

Keep components small, reusable, and free of business logic.
