"""Spark session + table I/O that works the same on Databricks and on a laptop."""
import os

import pandas as pd

from . import config


def on_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def get_spark():
    """Return the active Spark session (Databricks) or start a local Delta-enabled one."""
    from pyspark.sql import SparkSession

    active = SparkSession.getActiveSession()
    if active is None and not on_databricks():
        from delta import configure_spark_with_delta_pip

        local = config.LOCAL_RUNTIME_DIR
        local.mkdir(parents=True, exist_ok=True)
        builder = (
            SparkSession.builder.master("local[*]").appName("revenue_forecasting")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .config("spark.sql.warehouse.dir", str(local / "warehouse"))
            .config("spark.hadoop.javax.jdo.option.ConnectionURL",
                    f"jdbc:derby:;databaseName={local / 'metastore_db'};create=true")
            .config("spark.driver.memory", "4g")
            .config("spark.driver.extraJavaOptions", "-XX:ReservedCodeCacheSize=512m")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.ui.showConsoleProgress", "false")
            .enableHiveSupport()
        )
        active = configure_spark_with_delta_pip(builder).getOrCreate()
        active.sparkContext.setLogLevel("ERROR")
    else:
        active = active or SparkSession.builder.getOrCreate()

    active.sql(f"CREATE SCHEMA IF NOT EXISTS {config.SCHEMA}")
    active.sql(f"USE {config.SCHEMA}")
    return active


def write_table(spark, df, name: str):
    """Overwrite a Delta table from a Spark or pandas DataFrame and return the Spark DataFrame."""
    if isinstance(df, pd.DataFrame):
        df = spark.createDataFrame(df)
    (df.write.format("delta").mode("overwrite")
       .option("overwriteSchema", "true").saveAsTable(name))
    return spark.table(name)


def export_csv(spark, name: str) -> str:
    """Export a gold table to data/gold/<name>.csv (the files the Tableau workbook reads)."""
    config.GOLD_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.GOLD_EXPORT_DIR / f"{name}.csv"
    df = spark.table(name).toPandas()
    keys = [c for c in ("scenario", "strategy", "model", "month", "ds", "accounts_contacted") if c in df.columns]
    if keys:
        df = df.sort_values(keys)
    elif "revenue_risk" in df.columns:
        df = df.sort_values("revenue_risk", ascending=False)
    df.to_csv(path, index=False)
    return str(path)


def setup_mlflow():
    """Point MLflow at the workspace tracking server on Databricks, or a local file store otherwise."""
    import mlflow

    if on_databricks():
        user = get_spark().sql("SELECT current_user()").first()[0]
        mlflow.set_experiment(f"/Users/{user}/revenue_forecasting")
    else:
        config.LOCAL_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{config.LOCAL_RUNTIME_DIR / 'mlflow.db'}")
        if mlflow.get_experiment_by_name("revenue_forecasting") is None:
            mlflow.create_experiment("revenue_forecasting",
                                     artifact_location=(config.LOCAL_RUNTIME_DIR / "mlartifacts").as_uri())
        mlflow.set_experiment("revenue_forecasting")
    return mlflow
