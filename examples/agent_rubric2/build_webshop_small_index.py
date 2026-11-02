#!/usr/bin/env python3
"""Build the JSONL resource consumed by WebShop's 1k Lucene index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--webshop-root", type=Path, required=True)
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--attributes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    webshop_root = args.webshop_root.expanduser().resolve()
    products_path = args.products.expanduser().resolve()
    attributes_path = args.attributes.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    source_products = json.loads(products_path.read_text(encoding="utf-8"))
    if len(source_products) != 1000:
        raise ValueError(
            f"WebShop-small requires exactly 1000 source products, got {len(source_products)}: {products_path}"
        )

    # Reuse the official product normalization so the Lucene documents match
    # WebShop's own convert_product_file_format.py. Override the imported
    # defaults before loading engine.py so this remains correct even when the
    # checkout's utils.py was previously changed to the full catalog.
    sys.path.insert(0, str(webshop_root))
    import web_agent_site.utils as webshop_utils

    webshop_utils.DEFAULT_FILE_PATH = str(products_path)
    webshop_utils.DEFAULT_ATTR_PATH = str(attributes_path)
    from web_agent_site.engine.engine import load_products

    products, *_ = load_products(
        filepath=str(products_path),
        num_products=1000,
        human_goals=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    documents = []
    for product in products:
        option_texts = []
        for option_name, option_contents in product.get("options", {}).items():
            option_texts.append(f"{option_name}: {', '.join(option_contents)}")
        contents = " ".join(
            [
                product["Title"],
                product["Description"],
                product["BulletPoints"][0],
                ", and ".join(option_texts),
            ]
        ).lower()
        documents.append({"id": product["asin"], "contents": contents, "product": product})

    with output_path.open("w", encoding="utf-8") as output_file:
        for document in documents:
            output_file.write(json.dumps(document, ensure_ascii=False) + "\n")

    print(f"Prepared {len(documents)} WebShop-small search documents: {output_path}")


if __name__ == "__main__":
    main()
