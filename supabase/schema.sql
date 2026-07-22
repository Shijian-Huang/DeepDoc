create table if not exists public.analyses (
  analysis_id text primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  filename text,
  paper_title text,
  summary_mode text,
  processing_seconds numeric,
  summary text,
  created_at timestamptz not null default now(),
  record jsonb not null
);

create index if not exists analyses_user_created_at_idx
  on public.analyses (user_id, created_at desc);

alter table public.analyses enable row level security;

create policy "Users can read own analyses"
  on public.analyses
  for select
  using ((select auth.uid()) = user_id);

create policy "Users can insert own analyses"
  on public.analyses
  for insert
  with check ((select auth.uid()) = user_id);

create policy "Users can update own analyses"
  on public.analyses
  for update
  using ((select auth.uid()) = user_id)
  with check ((select auth.uid()) = user_id);

create policy "Users can delete own analyses"
  on public.analyses
  for delete
  using ((select auth.uid()) = user_id);

grant select, insert, update, delete on public.analyses to authenticated;
