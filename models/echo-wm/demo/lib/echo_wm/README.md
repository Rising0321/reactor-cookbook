# Schema-derived model client

`client.ts` and `client.react.tsx` mirror the model contract served by
`/schema`. Command parameters, model messages, and the `main_video` and
`main_audio` receive tracks are represented as TypeScript types.

`client.ts` contains the standalone client surface. `client.react.tsx`
contains `useEchoWmFlash()`, one hook per message, and the audiovisual
`EchoWmFlashMainVideoView` component. Both use
`@reactor-team/js-sdk` directly.

The app wraps the SDK's `ReactorProvider` rather than the model-specific
provider so `REACTOR_MODEL_NAME` can select the deployed model without changing
source code.

## Keeping it in sync

After changing the Python schema, start the model and save the served contract:

```sh
curl -fsS http://localhost:8080/schema -o /tmp/echo-wm-schema.json
```

Update both client files to match that contract, then verify the frontend:

```sh
pnpm typecheck
pnpm build
```
