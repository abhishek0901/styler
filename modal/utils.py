def download_classification_model(repo_id, local_dir):
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        ignore_patterns=["*.pt", "*.bin"],
        local_dir=local_dir,
    )


CATEGORY_MAP = {
    "ACTIVE ACCESSORIES": "Other",
    "ACTIVEWEAR": "Other",
    "ALL-IN-ONES": "Other",
    "BACKPACKS": "Other",
    "BEACH ACCESSORIES": "Other",
    "BEACHWEAR": "Other",
    "BELTS": "Other",
    "BLAZERS": "Top",
    "BOOTS": "Other",
    "BOW TIES": "Other",
    "BRIEFCASES": "Other",
    "BUCKET BAGS": "Other",
    "CANDLES AND HOME FRAGRANCE": "Other",
    "CARDHOLDERS": "Other",
    "CLUTCH BAGS": "Other",
    "COATS": "Top",
    "CROSS-BODY BAGS": "Other",
    "CUFFLINKS, PINS AND CLIPS": "Other",
    "CUSHIONS AND THROWS": "Other",
    "DENIM": "Top",
    "DRESSES": "Top",
    "ESPADRILLES": "Other",
    "FASHION JEWELRY": "Other",
    "FINE JEWELRY": "Other",
    "FLATS": "Other",
    "FORMAL SHOES": "Other",
    "GLASSES": "Other",
    "GLOVES": "Other",
    "HAIR ACCESSORIES": "Other",
    "HAND-HELD BAGS": "Other",
    "HATS": "Other",
    "HEELS": "Other",
    "HOME ACCESSORIES": "Other",
    "JACKETS": "Top",
    "JEANS": "Bottom",
    "JEWELRY": "Other",
    "KEY RINGS": "Other",
    "KNITWEAR": "Top",
    "LINGERIE AND NIGHTWEAR": "Other",
    "LOUNGEWEAR": "Bottom",
    "LUGGAGE AND TRAVEL BAGS": "Other",
    "MATCHING SETS": "Other",
    "MINI BAGS": "Other",
    "PANTS": "Top",
    "POCKET SQUARES": "Other",
    "POLO SHIRTS": "Top",
    "SANDALS": "Other",
    "SCARVES": "Other",
    "SHIRTS": "Top",
    "SHORTS": "Bottom",
    "SHOULDER BAGS": "Other",
    "SKIRTS": "Top",
    "SLIPPERS": "Other",
    "SNEAKERS": "Other",
    "SOCKS": "Other",
    "SUITS": "Top",
    "SUNGLASSES": "Other",
    "SWEATS": "Bottom",
    "SWIMWEAR": "Other",
    "T-SHIRTS": "Top",
    "TABLETOP": "Other",
    "TECHNOLOGY": "Other",
    "TIES": "Other",
    "TOP-HANDLE BAGS": "Other",
    "TOPS": "Top",
    "TOTE BAGS": "Other",
    "TRAVEL ACCESSORIES": "Other",
    "TRAVEL BAGS": "Other",
    "TROUSERS": "Bottom",
    "UNDERWEAR AND NIGHTWEAR": "Other",
    "WALLETS": "Other",
    "WASH BAGS": "Other",
    "WATCHES": "Other",
}
