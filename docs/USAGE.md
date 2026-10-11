# vsort usage guide

For an overview of the project, see the [README](../README.md).

## Principles

- **Dry run by default**: any command that would touch files only writes a plan (CSV, opens in Excel) unless `--apply` is given.
- **Every action can be undone**: applied runs leave a plain-text journal in `Workspace\_logs`; `vsort undo` replays it in reverse.
- **Never overwrite**: if the destination name exists, ` (2)` is appended; a name that appears only at execution time makes that single item fail and leaves the source untouched.
- **Resumable**: after a crash or a closed window, `vsort resume` continues where it stopped. No new run starts while an interrupted one exists.
- **Safe across drives**: copy to a temp file, verify size and content, rename, and only then delete the source.

## Folders

```
Workspace\
  01_Inbox\            videos you picked for processing
  02_Parsed_Staging\   parsed, waiting to be routed (or to retry a failed lookup)
  03_AI_Workspace\
    Identify_Queue\    not recognized yet, waiting for AI identification
    Refine_Queue\      partly known, waiting for AI to fill in the rest
  04_Final_Library\    finished library (can live on any drive via library_dir in config.yaml)
  05_Review\           waiting for human review
  _logs\               run journals (used for undo)
  _reports\            dry-run plans
```

With `root_dir` set in `config.yaml` (or via "Change…" next to the sorting location on the web home page), for example `D:\Videos`, the layout becomes `D:\Videos\sorting\01_Inbox … 05_Review` for videos being sorted and `D:\Videos\<Creator>\…` for the finished library. Without it, `workspace_dir` and `library_dir` are used.

## Commands

Run from the project folder with `vsort.bat` (or drag videos onto `drop_videos_here.bat`):

```bat
vsort.bat web                          :: web home page: pick videos or folders -> sort -> review (the easiest way; open_web.bat does the same)
vsort.bat init                         :: create the folders
vsort.bat ingest D:\Downloads\folder   :: dry run: show how the files would enter 01_Inbox
vsort.bat ingest --pick                :: pick video files in a dialog (--pick-folder for folders)
vsort.bat apply                        :: apply the plan from the last dry run
vsort.bat ingest file1 file2 --apply   :: dry run and apply in one step
vsort.bat ingest ... --copy            :: copy instead of move, keeping the originals
vsort.bat process                      :: dry run: parse filenames in 01_Inbox, look up sources, route to 03
vsort.bat process --offline            :: no network, filenames only
vsort.bat process --apply              :: apply parsing and routing
vsort.bat inspect D:\Videos --csv r.csv :: show parse results for any folder without moving files
vsort.bat analyze                      :: dry run: AI analysis of 03 (confident -> 04, uncertain -> 05)
vsort.bat analyze --apply              :: apply analysis and filing
vsort.bat analyze --library            :: also re-analyze library videos with missing category or tags
vsort.bat review --web                 :: review page only (preview frames, edit fields, confirm to file)
vsort.bat review                       :: or write 05_Review\review.csv for review in Excel
vsort.bat review --import --apply      :: import the decisions filled in Excel
vsort.bat dupes                        :: list duplicate videos (never deletes)
vsort.bat label D:\Videos --n 150      :: labeling page: tag sampled videos with animation style (MMD / 3D / 2D, multi-select)
vsort.bat label D:\Videos --n 100 --category-uncertain :: pick the 100 videos whose category is least certain
vsort.bat train-style                  :: compare CLIP / SigLIP / a small classifier and train the best one
vsort.bat train-category               :: retrain the category classifier on your labels (tags + Qwen3-VL + audio)
vsort.bat authors                      :: update the creator alias table (runs automatically after process --apply)
vsort.bat authors --accept             :: apply the Y/N decisions you filled in authors_pending.csv
vsort.bat undo                         :: dry run: undo the last run
vsort.bat undo --apply                 :: undo it for real
vsort.bat batches                      :: list import batches: source, count, where each video is now
vsort.bat undo --batch batch-ID --apply :: undo a whole batch, including later routing, filing and review
vsort.bat rematch [--apply]            :: re-match unknown-creator videos against known creators by style fingerprint
vsort.bat resume                       :: resume an interrupted run or retry failed items
vsort.bat runs                         :: list run journals
vsort.bat status                       :: video counts per folder
vsort.bat --library D:\Library ...     :: use a different library location for this run
```

All other settings are in `config.yaml` and take effect on the next run. Folders listed in `skip_folders` are skipped entirely when sampling for labeling and re-evaluation.

## Filenames and dictionaries

- Final name: `Title [Creator][Work][CharacterA,CharacterB][Style][Category].mp4`; missing fields are `[-]`.
  Example: `New Year Dance [SomeCreator][Genshin Impact][Raiden Shogun][MMD][Dance].mp4`
- Character names in filenames use the English name column of `dict\characters.csv`, at most 3. Chinese and Japanese names and the full list are kept in the sidecar (`characters_i18n`). Characters marked male in the dictionary (or tagged `male_*` by the model) are left out.
- One confident main character is enough to auto-file. Uncertain characters go to `characters_unconfirmed` in the sidecar with `characters_incomplete: true` for later filtering. The work does not block auto-filing; the most likely candidate is used.
- Creator: by default, the outermost Inbox folder that is not a generic name (unsorted, site names, dates, plain numbers…).
- Works and characters are matched against `dict\works.csv` and `dict\characters.csv` (edit in Excel, save as CSV UTF-8). Adding a row teaches the parser a new character or alias.
- Every video gets a `<video name>.vsort.json` sidecar with its original full path (`original_path`), batch ID (`batch`), original filename, and the parse and AI results. It moves with the video.

## Source lookup and creators

- When a filename or folder carries a post ID from a supported creator platform, the title, creator and tags are looked up on that site, falling back from public data to archive mirrors to cookies. Results are cached in `Workspace\_cache`.
- Videos whose lookup failed because of the network stay in `02_Parsed_Staging` and are retried by the next `process`.
- Logins and cookies go in `secrets.yaml` (see `secrets.example.yaml`; git-ignored). Public data works without them. **Put a space after each colon**, e.g. `api_key: abc123`; without it the program warns and tries to split the value, but please fix the file.
- When a filename has no site ID, the source is searched in this order: the download manager's history (matched by file size and name) -> filename hints (creator folder + title + date) on the supported sites -> title search on a video index site. Each source can be turned off under `lookup:` (`use_idm`, `use_hints`, `use_oreno3d`). The first hint scan of a creator takes 2–3 minutes; later scans hit the cache.
- Creators are identified by account ID (`dict\authors.csv`). An unseen account whose name is unlike every known creator is added automatically. One that resembles a known creator, or sits in a known creator's folder, goes to `dict\authors_pending.csv` for your Y/N.

## AI analysis

- Everything runs on the local GPU and videos are never uploaded. Models are set under `ai:` in `config.yaml` and downloaded to `models\`.
  - Characters and visual tags (`ai.model`): a WD-style anime tagger such as `pixai-labs/pixai-tagger-v1.0`, `SmilingWolf/wd-eva02-large-tagger-v3` or `Camais03/camie-tagger-v2`; compare them with `tools\compare_taggers.py`.
  - Animation style (`ai.style_model`, multi-label such as `[MMD,3D]`): CLIP zero-shot, `google/siglip2-so400m-patch14-384` (SigLIP zero-shot), or `trained` (a small classifier trained on samples labeled with `vsort label`).
- Category (Dance / Mix / Scene): a classifier trained on your labels, using tag ratios, Qwen3-VL per-frame and video-level judgments, and audio features (`vsort\audio.py`: tempo, vocal presence, timbre; no model download). A video counts as certain when its top category probability is at least `category_sure_prob` (0.7). Only uncertain videos get the second-stage prompt (`category_second_stage`).
- What counts as Scene is defined locally, not in code: the tagger labels in `dict\scene_tags.txt` and the prompts in `prompts\*.txt`. Each falls back to the shipped `*.example.txt` file when no local version exists.
- Creator prior (`author_prior_weight`): category probabilities are multiplied by the creator's past category distribution (from labeled videos plus filed and reviewed library videos), so accuracy improves as a creator's catalogue grows. Set to 0 to disable.
- Only certain videos go to 04. These go to 05 for review, each with a reason and preview frames: category near the threshold, filename and AI disagree, low style confidence, unknown creator, or unknown character or work.
- If a creator has 3 or more videos (labeled data plus library) and at least 80% share one animation style, new videos inherit that style. A style probability of 0.1 or less counts as clearly absent and does not trigger review.
- With an unknown creator, similar-style creators are recorded for reference only, and the video is filed under `作者未知` (unknown creator); leaving the creator empty during review files it there too.
- `作者未知` is not a dead end: after each batch (or via "Re-match unknown creators" on the home page, or `vsort rematch`), style fingerprints are compared with the currently known creators. Videos with similarity ≥ 0.5 and a lead of ≥ 0.1 over the runner-up go back to 05 with the creator pre-filled. Once one video of a creator is identified, earlier unidentified ones can be recovered (421-video test: 98% correct at this threshold).
- Reverse image search (SauceNAO) is off by default and needs an API key; see `secrets.example.yaml`.
- Analysis results are cached in `Workspace\_cache\ai`, so applying after a dry run does not recompute.

## Development

```bat
.venv\Scripts\python.exe -m pytest -q
```

The venv is created with `--system-site-packages` to reuse an existing CUDA PyTorch install instead of downloading another copy.
