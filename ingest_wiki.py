import os
from utils import (
    read_markdown_file,
    clean_markdown,
    create_chunks,
    normalize_text,
    store_article,
)

def main():
    md_dir = "wiki"
    md_files = sorted(
        os.path.join(md_dir, f)
        for f in os.listdir(md_dir)
        if f.endswith(".md")
    )

    if not md_files:
        print("No markdown files found in", md_dir)
        return

    total = 0

    for path in md_files:
        content = read_markdown_file(path)
        content = clean_markdown(content)
        chunks = create_chunks(content)
        chunks = [normalize_text(c) for c in chunks]

        if chunks:
            store_article(chunks, source=path)
            total += len(chunks)

        print(f"{path}: stored {len(chunks)} chunks")

    print(f"Total chunks stored: {total}")


if __name__ == "__main__":
    main()
