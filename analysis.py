import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/vgsales.csv"
CHARTS_DIR = "charts"


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset: {path}")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with missing critical fields
    df = df.dropna(subset=["Name", "Platform", "Year", "Genre", "Publisher", "Global_Sales"]).copy()

    # Convert Year to int
    df["Year"] = df["Year"].astype(int)

    # Ensure sales columns are numeric
    sales_cols = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]
    for col in sales_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=sales_cols)
    return df


def save_barh(series: pd.Series, title: str, xlabel: str, filename: str, top_n: int = 10):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    s = series.sort_values(ascending=False).head(top_n)[::-1]  # reverse for nice horizontal bars

    plt.figure(figsize=(10, 6))
    plt.barh(s.index.astype(str), s.values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    out_path = os.path.join(CHARTS_DIR, filename)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved chart: {out_path}")


def main():
    os.makedirs(CHARTS_DIR, exist_ok=True)

    df = load_data(DATA_PATH)
    df = clean_data(df)

    print("Rows after cleaning:", len(df))
    print("\nTop 5 rows:\n", df.head())

    # 1) Top genres
    genre_sales = df.groupby("Genre")["Global_Sales"].sum()
    save_barh(
        genre_sales,
        title="Top Genres by Global Sales",
        xlabel="Global Sales (Millions)",
        filename="genre_sales.png",
        top_n=10
    )

    # 2) Top platforms
    platform_sales = df.groupby("Platform")["Global_Sales"].sum()
    save_barh(
        platform_sales,
        title="Top Platforms by Global Sales",
        xlabel="Global Sales (Millions)",
        filename="platform_sales.png",
        top_n=10
    )

    # 3) Yearly trend
    yearly_sales = df.groupby("Year")["Global_Sales"].sum().sort_index()

    plt.figure(figsize=(10, 6))
    plt.plot(yearly_sales.index, yearly_sales.values)
    plt.title("Global Video Game Sales Over Time")
    plt.xlabel("Year")
    plt.ylabel("Global Sales (Millions)")
    plt.tight_layout()
    out_path = os.path.join(CHARTS_DIR, "yearly_sales_trend.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved chart: {out_path}")

    # Peak year callout
    peak_year = yearly_sales.idxmax()
    peak_sales = yearly_sales.max()
    print(f"\nMarket Peak: {peak_year} with {peak_sales:.2f} million global sales")

    # 4) Top publishers
    publisher_sales = df.groupby("Publisher")["Global_Sales"].sum()
    save_barh(
        publisher_sales,
        title="Top Publishers by Global Sales",
        xlabel="Global Sales (Millions)",
        filename="publisher_sales.png",
        top_n=10
    )

    # 5) Top 10 best-selling games
    top_games = df.sort_values("Global_Sales", ascending=False).head(10)
    print("\nTop 10 Best-Selling Games:")
    print(top_games[["Name", "Platform", "Year", "Genre", "Global_Sales"]])

    # Save top games as a CSV artifact (nice for submissions)
    top_games.to_csv(os.path.join(CHARTS_DIR, "top_10_games.csv"), index=False)
    print("Saved table: charts/top_10_games.csv")

    # 6) Regional market totals + chart
    regions = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
    region_totals = df[regions].sum()

    plt.figure(figsize=(8, 6))
    region_totals.plot(kind="bar")
    plt.title("Sales by Region")
    plt.ylabel("Sales (Millions)")
    plt.tight_layout()
    out_path = os.path.join(CHARTS_DIR, "region_sales.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved chart: {out_path}")

    print("\nRegional Sales Totals:")
    print(region_totals)

    # 7) Genre popularity over time (top 5 genres for readability)
    genre_year = df.groupby(["Year", "Genre"])["Global_Sales"].sum().unstack().fillna(0)
    top_genres = genre_year.sum().sort_values(ascending=False).head(5).index
    genre_year_top = genre_year[top_genres]

    plt.figure(figsize=(12, 7))
    genre_year_top.plot()
    plt.title("Top 5 Genres: Global Sales Over Time")
    plt.ylabel("Global Sales (Millions)")
    plt.tight_layout()
    out_path = os.path.join(CHARTS_DIR, "genre_trends_top5.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved chart: {out_path}")

    # 8) Platform lifecycle for top 5 platforms
    platform_year = df.groupby(["Year", "Platform"])["Global_Sales"].sum().unstack().fillna(0)
    top_platforms = platform_year.sum().sort_values(ascending=False).head(5).index

    plt.figure(figsize=(12, 7))
    platform_year[top_platforms].plot()
    plt.title("Top 5 Platforms: Global Sales Over Time")
    plt.ylabel("Global Sales (Millions)")
    plt.tight_layout()
    out_path = os.path.join(CHARTS_DIR, "platform_lifecycle_top5.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved chart: {out_path}")

    # 9) Correlation matrix
    corr = df[["NA_Sales", "EU_Sales", "JP_Sales", "Global_Sales"]].corr()
    print("\nSales Correlation Matrix:")
    print(corr)

    # Quick insights
    print("\nQuick Insights:")
    print("Top Genre:", genre_sales.idxmax())
    print("Top Platform:", platform_sales.idxmax())
    print("Best Sales Year:", yearly_sales.idxmax())
    print("Top Publisher:", publisher_sales.idxmax())


if __name__ == "__main__":
    main()