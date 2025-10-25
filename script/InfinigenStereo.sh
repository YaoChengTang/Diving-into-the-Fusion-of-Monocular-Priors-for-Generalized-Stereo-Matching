
# hf download pvl-lab/InfinigenStereo --repo-type dataset --local-dir datasets/InfinigenStereo

cd datasets/InfinigenStereo/release_full


# find . -type f -name "*.tar.gz" -print0 | while IFS= read -r -d '' f; do
#   dir=$(dirname "$f")
#   file=$(basename "$f")              # 例如: ff760b.tar.gz
#   name="${file%.tar.gz}"             # 去掉 .tar.gz -> ff760b
#   target="$dir/$name"                # 目标目录: .../ff760b

#   mkdir -p "$target"
#   echo "解压: $f -> $target"
#   tar -xzf "$f" -C "$target"
# done


# find . -type f -name "*.tar.gz" | head -n 3 | while read -r f; do
#   dir=$(dirname "$f")
#   file=$(basename "$f")
#   name="${file%.tar.gz}"
#   target="$dir/$name"

#   mkdir -p "$target"
#   echo "解压: $f -> $target"
#   tar -xzf "$f" -C "$target"
# done


find . -type f -name "*.tar.gz" | parallel '
  dir=$(dirname {});
  file=$(basename {});
  name=${file%.tar.gz};
  target="$dir/$name";
  mkdir -p "$target";
  echo "解压: {} -> $target";
  tar -xzf {} -C "$target"
'

# find . -type f -name "*.tar.gz" | head -n 3 | parallel '
#   dir=$(dirname {});
#   file=$(basename {});
#   name=${file%.tar.gz};
#   target="$dir/$name";
#   mkdir -p "$target";
#   echo "解压: {} -> $target";
#   tar -xzf {} -C "$target"
# '