import inngest

ANALYZE_EVENT = "autoinsight/dataset.analyze"

inngest_client = inngest.Inngest(
    app_id="autoinsight",
    is_production=False,  # local dev server; revisit when you deploy
)