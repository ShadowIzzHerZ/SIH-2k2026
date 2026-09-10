# SIH26168: Supabase User Accounts & Fleet Registry

## Why this exists

A judge asked: *"what if 1000+ people join the app?"* The on-device dead-reckoning
pipeline ([app/](app/), [src/](src/)) needs no backend at all — it's the answer to
that question for the core navigation feature. But **user accounts and a
fleet/control-room view** are a different concern, and that's what this piece answers:
a managed, autoscaling Postgres database (Supabase) handling sign-up/login and device
ownership, separate from the high-frequency 10Hz telemetry service
([Sih-Backend](Sih-Backend/), being migrated to AWS Lambda/DynamoDB — see
[Sih-Backend/AMPLIFY_MIGRATION.md](Sih-Backend/AMPLIFY_MIGRATION.md)).

**Why two databases, not one**: user accounts (sign-up, profile, "which device does
this person own") are low-frequency, relational, and exactly what Postgres+Auth is
built for. Device telemetry (10Hz IMU/position streams from potentially many
concurrent vehicles) is high-frequency, single-item-keyed, bursty — DynamoDB's native
shape. Using one store for both would mean either running the telemetry hot-path
through a service not designed for that write pattern, or bolting real user auth onto
DynamoDB by hand. Splitting them is the more scalable answer, not extra complexity for
its own sake.

## What's set up

**Supabase project**: `sih26168-dead-reckoning` (ref `iekkwudnpvijroqvknwq`, org `Zen`,
region `ap-south-1`). Deliberately a **new, dedicated project** — the org's existing
"Hackathon" project already holds an unrelated civic-issue-reporting app's live schema
and data; this keeps SIH26168 isolated from that.

- URL: `https://iekkwudnpvijroqvknwq.supabase.co`
- Publishable (anon) key: safe to embed in the Android app's client code —
  Supabase's publishable keys are meant to be public; Row Level Security (RLS, see
  below) is what actually enforces access control, not keeping this key secret.
  `sb_publishable_g0HHEKwCxphZyVWAFvYqWw_RBATOdJ7`

### Schema

- **`profiles`** — one row per signed-up user, 1:1 with Supabase's built-in
  `auth.users` (auto-created via an `on_auth_user_created` trigger on signup — the
  app never has to remember to create this row itself). Carries `role`
  (`driver`/`admin` — `admin` is what would gate a future control-room dashboard's
  access to see every device, not just your own) and
  `data_contribution_opt_in` (explicit, defaults-false consent for the
  record-my-drives-to-help-train-the-model feature — see the "Auto-recording for
  retraining" discussion this was designed alongside).
- **`devices`** — links a phone/vehicle's `device_id` (the same identifier used as
  the JWT `sub`/`device_id` claim and DynamoDB partition key on the AWS backend side)
  to the `profiles` row that registered it.
- **`drive_sessions`** — a catalog of trips: start/end time, start/end GPS
  coordinates (the ground-truth bookends around a blackout that the training
  pipeline already uses to compute end-to-end drift %, matching
  [src/data/windowing.py](src/data/windowing.py)'s methodology), whether it had a
  blackout, a pointer (`raw_data_ref`) to wherever the actual raw IMU log lives
  (S3/DynamoDB, not this table), and `contribute_to_training` — a **per-session
  snapshot** of the user's consent at the time, not a live join to `profiles`, so
  revoking consent later doesn't retroactively misrepresent what a past session was
  collected under.

All three tables have **Row Level Security enabled**: a user can only read/write
their own profile, their own devices, and sessions for devices they own; an `admin`
role can read across all of them (the fleet view's access path). Verified clean via
Supabase's security advisor after one fix (a trigger-only function's PostgREST RPC
execute grant was revoked — it was reachable by `anon`/`authenticated` roles by
default, though it would have errored if actually called that way since trigger
functions require trigger context).

## What's NOT built yet

This is the account/consent layer only. Still outstanding, and each is real,
separate scope:
1. **Android app changes**: no code yet calls Supabase Auth for sign-up/login, and
   no code yet writes to `devices`/`drive_sessions`. The app today has zero network
   dependency for its core dead-reckoning pipeline (by design) — adding this means
   deciding how much of the account/upload flow is required vs. optional-when-online.
2. **AWS backend deployment**: [Sih-Backend](Sih-Backend/)'s Amplify/Lambda/DynamoDB
   migration is code-complete through Phase 3 but not yet deployed — needs AWS
   credentials connected, still pending.
3. **Fleet/control-room dashboard**: nothing built yet — would read `admin`-role
   data from both Supabase (who owns which device) and the AWS backend
   (where each device currently is).
4. **Retraining pipeline**: `drive_sessions` is the catalog/index; there's no job yet
   that actually pulls opted-in raw sessions, validates them, and retrains/redeploys
   the model.
