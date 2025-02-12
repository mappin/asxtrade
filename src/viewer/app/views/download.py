"""
Responsible for implementing download of datasets (CSV/Excel/TSV/Parquet)
on various pages for the user
"""
import tempfile
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from app.models import cached_all_stocks_cip, Timeframe, validate_user


def save_dataframe_to_file(df, filename, output_format):
    assert output_format in ("csv", "excel", "tsv", "parquet")
    assert df is not None and len(df) > 0
    assert len(filename) > 0

    if output_format == "csv":
        df.to_csv(filename)
        return "text/csv"
    elif output_format == "excel":
        df.to_excel(filename)
        return "application/vnd.ms-excel"
    elif output_format == "tsv":
        df.to_csv(filename, sep="\t")
        return "text/tab-separated-values"
    elif output_format == "parquet":
        df.to_parquet(filename)
        return "application/octet-stream"  # for now, but must be something better...
    else:
        raise ValueError("Unsupported format {}".format(output_format))


def get_dataset(dataset_wanted):
    assert dataset_wanted in set(["market_sentiment"])

    if dataset_wanted == "market_sentiment":
        df = cached_all_stocks_cip(Timeframe())
        return df
    else:
        raise ValueError("Unsupported dataset {}".format(dataset_wanted))


@login_required
def download_data(request, dataset=None, output_format="csv"):
    validate_user(request.user)

    with tempfile.NamedTemporaryFile() as fh:
        df = get_dataset(dataset)
        content_type = save_dataframe_to_file(df, fh.name, output_format)
        fh.seek(0)
        response = HttpResponse(fh.read(), content_type=content_type)
        response["Content-Disposition"] = "inline; filename=temp.{}".format(output_format)
        return response
