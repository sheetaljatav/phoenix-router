# Publish & Submit — exact steps

## 1. One-time: GitHub token (you do this, ~2 min)

The image will live on GitHub Container Registry (ghcr.io) and the code in a
public GitHub repo — one account, one token.

1. Open https://github.com/settings/tokens/new
2. Note: `amd-hackathon`; Expiration: 30 days
3. Tick scopes: **repo** and **write:packages**
4. Generate, copy the `ghp_...` token

## 2. Create the repo (you, ~1 min)

1. https://github.com/new → name: `phoenix-router`, Public, no README
2. Create.

## 3. Push code + image (Claude runs these once you paste the token)

```powershell
# code
git remote add origin https://github.com/sheetaljatav/phoenix-router.git
git push -u origin main

# image
docker login ghcr.io -u sheetaljatav        # password = the token
docker tag phoenix-router:latest ghcr.io/sheetaljatav/phoenix-router:latest
docker push ghcr.io/sheetaljatav/phoenix-router:latest
```

Then make the package public (required so judges can pull it):
https://github.com/users/sheetaljatav/packages/container/phoenix-router/settings
→ Danger Zone → Change visibility → Public.

## 4. Submit on lablab.ai

On the hackathon page → your team → Submit:

- Title / descriptions: copy from `SUBMISSION.md`
- Cover image: `assets/cover.png`
- Slides: `assets/Phoenix-Router-deck.pdf` (or .pptx)
- Video: record 1-2 min (screen-record the README + a container run;
  phone camera or OBS both fine — judges only need to see it working)
- GitHub repo: https://github.com/sheetaljatav/phoenix-router
- Docker image (the field the evaluator pulls):
  `ghcr.io/sheetaljatav/phoenix-router:latest`

Deadline: **July 11** (check the Event Schedule tab for local-time cutoff).
Submissions are rate-limited to 10/hour, and the leaderboard shows results —
submit early, keep the last slot for a final image.
