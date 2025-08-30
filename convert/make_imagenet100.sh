#!/usr/bin/env bash
# Create an ImageNet-100 subset from an ImageNet-1k ImageFolder.
#
# Usage examples:
#   # Random 100 classes with symlinks
#   bash convert/make_imagenet100.sh --src /path/imagenet --dst /path/imagenet-100
#
#   # Use a class list file (one class per line)
#   bash convert/make_imagenet100.sh --src /path/imagenet --dst /path/imagenet-100 --classes classes_100.txt
#
#   # Hardlinks (distributed-friendly) or copy
#   bash convert/make_imagenet100.sh --src /path/imagenet --dst /path/imagenet-100 --mode hardlink
#   bash convert/make_imagenet100.sh --src /path/imagenet --dst /path/imagenet-100 --mode copy
#
# Notes:
# - Source must be in ImageFolder layout:
#     <src>/train/<class>/* and <src>/(val|validation)/<class>/*
# - Selected classes are saved to <dst>/classes_selected.txt

set -uo pipefail

SRC=""
DST=""
K=100
CLASSES_FILE=""
MODE="symlink"       # symlink|hardlink|copy
MAX_PER_CLASS=""     # optional
SEED=""             # optional, used for deterministic sampling (requires python)
DRY_RUN=0

log()  { echo "[INFO]  $*"; }
warn() { echo "[WARN]  $*" 1>&2; }
err()  { echo "[ERROR] $*" 1>&2; exit 1; }

usage() {
  cat <<EOF
Make ImageNet-100 subset from ImageNet-1k ImageFolder

Required:
  --src PATH             Source ImageNet root (contains train/ and val|validation/)
  --dst PATH             Destination root for ImageNet-100 subset

Optional:
  --k INT                Number of classes to keep (default: 100)
  --classes FILE         File with class folder names (one per line). If not set, random sample.
  --mode MODE            symlink | hardlink | copy (default: symlink)
  --max-per-class INT    Limit images per class per split (default: all)
  --seed INT             Seed for deterministic random sampling (requires python)
  --dry-run              Print actions without writing files
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) SRC="$2"; shift 2;;
    --dst) DST="$2"; shift 2;;
    --k) K="$2"; shift 2;;
    --classes) CLASSES_FILE="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    --max-per-class) MAX_PER_CLASS="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) warn "Unknown argument: $1"; usage; exit 1;;
  esac
done

[[ -z "$SRC" ]] && err "--src is required"
[[ -z "$DST" ]] && err "--dst is required"

case "$MODE" in
  symlink|hardlink|copy) ;;
  *) err "--mode must be one of: symlink|hardlink|copy";;
esac

# Resolve splits
resolve_split() {
  local root="$1"; local name="$2"
  if [[ "$name" == "train" ]]; then
    [[ -d "$root/train" ]] && echo "$root/train" && return 0
    echo ""; return 0
  fi
  # val / validation
  if [[ -d "$root/val" ]]; then echo "$root/val"; return 0; fi
  if [[ -d "$root/validation" ]]; then echo "$root/validation"; return 0; fi
  echo ""; return 0
}

is_split_structured() {
  local split_root="$1"
  [[ -z "$split_root" ]] && return 1
  [[ ! -d "$split_root" ]] && return 1
  # Structured if there is at least one subdirectory
  local d
  for d in "$split_root"/*; do
    [[ -d "$d" ]] && return 0
  done
  return 1
}

list_class_dirs() {
  local split_root="$1"
  local d
  for d in "$split_root"/*; do
    [[ -d "$d" ]] || continue
    local b
    b=$(basename "$d")
    [[ "$b" == .* ]] && continue
    echo "$b"
  done | LC_ALL=C sort
}

read_class_list_file() {
  local file="$1"
  [[ -f "$file" ]] || err "Class list file not found: $file"
  grep -v '^#' "$file" | sed '/^$/d'
}

select_random_classes() {
  # Args: K all_classes...  Optional SEED
  local k="$1"; shift
  local seed="$1"; shift
  if [[ -n "$seed" ]]; then
    # Use python for deterministic sampling
    python3 - "$k" "$seed" "$@" <<'PY'
import os,random,sys
k = int(sys.argv[1]); seed = int(sys.argv[2]); classes = sys.argv[3:]
if not (0 < k <= len(classes)):
    print(f"[ERROR] --k must be within (0, {len(classes)}]", file=sys.stderr)
    sys.exit(1)
random.seed(seed)
sel = sorted(random.sample(classes, k))
print("\n".join(sel))
PY
    return
  fi
  # Fallback: best-effort with shuf if available (non-deterministic)
  if command -v shuf >/dev/null 2>&1; then
    printf "%s\n" "$@" | shuf -n "$k" | LC_ALL=C sort
  else
    # POSIX fallback: pseudo-random via awk + sort (non-deterministic)
    printf "%s\n" "$@" | awk 'BEGIN{srand()} {print rand()"\t"$0}' | sort -k1,1 | cut -f2- | head -n "$k" | LC_ALL=C sort
  fi
}

ensure_dir() { [[ "$DRY_RUN" -eq 1 ]] || mkdir -p "$1"; }

lower_ext() { echo "$1" | tr '[:upper:]' '[:lower:]'; }

is_image_file() {
  local f="$1"; local ext="${f##*.}"; ext=$(lower_ext "$ext")
  case "$ext" in
    jpg|jpeg|png|bmp|ppm|pgm|tif|tiff|webp|jpeg) return 0;;
    JPG|JPEG) return 0;;
    *) return 1;;
  esac
}

place_file() {
  local src="$1"; local dst="$2"
  if [[ -e "$dst" ]]; then return 0; fi
  if [[ "$DRY_RUN" -eq 1 ]]; then return 0; fi
  case "$MODE" in
    symlink)
      # Try creating a relative symlink (portable via python). Fallback to hardlink.
      local parent
      parent=$(dirname "$dst")
      local rel
      rel=$(python3 - "$src" "$parent" <<'PY'
import os,sys
src, parent = sys.argv[1], sys.argv[2]
print(os.path.relpath(src, start=parent))
PY
)
      if ln -s "$rel" "$dst" 2>/dev/null; then
        :
      else
        ln "$src" "$dst"
      fi
      ;;
    hardlink)
      ln "$src" "$dst" ;;
    copy)
      cp -p "$src" "$dst" ;;
  esac
}

populate_split() {
  # Args: SRC_SPLIT DST_SPLIT CLASSES_FILE_PATH
  local src_split="$1"; local dst_split="$2"; local classes_file="$3"
  local cls present=0 files=0
  [[ -z "$src_split" || ! -d "$src_split" ]] && echo "0 0" && return 0
  if ! is_split_structured "$src_split"; then
    warn "Split not organized by class directories: $src_split. Skipping this split."
    echo "0 0"; return 0
  fi
  while IFS= read -r cls; do
    [[ -z "$cls" ]] && continue
    local src_cls="$src_split/$cls"
    if [[ ! -d "$src_cls" ]]; then
      warn "Class not found in split: $cls (root=$src_split)"
      continue
    fi
    present=$((present+1))
    local dst_cls="$dst_split/$cls"
    ensure_dir "$dst_cls"
    local count=0
    local f
    for f in "$src_cls"/*; do
      [[ -f "$f" ]] || continue
      if ! is_image_file "$f"; then continue; fi
      local base
      base=$(basename "$f")
      place_file "$f" "$dst_cls/$base"
      files=$((files+1))
      count=$((count+1))
      if [[ -n "$MAX_PER_CLASS" && "$count" -ge "$MAX_PER_CLASS" ]]; then
        break
      fi
    done
  done < "$classes_file"
  echo "$present $files"
}

# Resolve roots
SRC=$(cd "$SRC" && pwd)
DST=$(mkdir -p "$DST" && cd "$DST" && pwd)

TRAIN_SRC=$(resolve_split "$SRC" train)
VAL_SRC=$(resolve_split "$SRC" val)
[[ -z "$TRAIN_SRC" ]] && err "Could not find train/ under: $SRC"

# Get class list from train
mapfile -t ALL_CLASSES < <(list_class_dirs "$TRAIN_SRC")
[[ ${#ALL_CLASSES[@]} -eq 0 ]] && err "No class directories found under train/: $TRAIN_SRC"

SELECTED_CLASSES_FILE="$DST/classes_selected.txt"
if [[ -n "$CLASSES_FILE" ]]; then
  log "Using class list file: $CLASSES_FILE"
  # Keep only classes present in train
  mapfile -t WANTED < <(read_class_list_file "$CLASSES_FILE")
  # Filter
  : > "$SELECTED_CLASSES_FILE"
  missing=0; kept=0
  for c in "${WANTED[@]}"; do
    if [[ -d "$TRAIN_SRC/$c" ]]; then
      echo "$c" >> "$SELECTED_CLASSES_FILE"; kept=$((kept+1))
    else
      missing=$((missing+1))
    fi
  done
  if [[ $missing -gt 0 ]]; then
    warn "$missing classes not found in train/. They were skipped."
  fi
else
  # Random sample of K classes
  if [[ "$K" -le 0 || "$K" -gt ${#ALL_CLASSES[@]} ]]; then
    err "--k must be within (0, ${#ALL_CLASSES[@]}]"
  fi
  log "Sampling $K classes from ${#ALL_CLASSES[@]} (seed=${SEED:-none})"
  if [[ -n "$SEED" ]]; then
    mapfile -t SEL < <(select_random_classes "$K" "$SEED" "${ALL_CLASSES[@]}")
  else
    mapfile -t SEL < <(select_random_classes "$K" "" "${ALL_CLASSES[@]}")
  fi
  printf "%s\n" "${SEL[@]}" > "$SELECTED_CLASSES_FILE"
fi

log "Selected classes file: $SELECTED_CLASSES_FILE"
[[ "$DRY_RUN" -eq 1 ]] && log "DRY-RUN enabled; no filesystem changes will be made."

# Create destination split roots
TRAIN_DST="$DST/train"; ensure_dir "$TRAIN_DST"
if [[ -n "$VAL_SRC" && -d "$VAL_SRC" ]] && is_split_structured "$VAL_SRC"; then
  VAL_DST="$DST/val"; ensure_dir "$VAL_DST"
else
  VAL_DST=""; [[ -n "$VAL_SRC" ]] || warn "No validation split found. Only train/ will be created."
fi

# Populate
read TRAIN_PRESENT TRAIN_FILES < <(populate_split "$TRAIN_SRC" "$TRAIN_DST" "$SELECTED_CLASSES_FILE")
VAL_PRESENT=0; VAL_FILES=0
if [[ -n "$VAL_DST" ]]; then
  read VAL_PRESENT VAL_FILES < <(populate_split "$VAL_SRC" "$VAL_DST" "$SELECTED_CLASSES_FILE")
fi

echo "[DONE] Summary:"
echo "  Train: classes=${TRAIN_PRESENT}/$(wc -l < "$SELECTED_CLASSES_FILE") files=${TRAIN_FILES} src=${TRAIN_SRC}"
if [[ -n "$VAL_DST" ]]; then
  echo "  Val  : classes=${VAL_PRESENT}/$(wc -l < "$SELECTED_CLASSES_FILE") files=${VAL_FILES} src=${VAL_SRC}"
fi
echo "  Output: $DST (mode=$MODE, max_per_class=${MAX_PER_CLASS:-all})"
