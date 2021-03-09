"""
Using django messages, apply end-user alerts and key information in relation to the currently
displayed page. Provides a simple API to other parts of the app.
"""
from django.contrib import messages


def info(request, msg):
    """
    Log the message to the current page template if request is not None. Log msg
    to stdout also.
    """
    assert len(msg) > 0
    if request is not None:
        messages.info(request, msg, extra_tags="alert alert-secondary", fail_silently=True)
    print(msg)

def warning(request, msg):
    """
    Log the message as a warning (orange) appearance on the page iff request is not None.
    Also log to stdout.
    """
    assert len(msg) > 0
    if request is not None:
        messages.warning(request, msg, extra_tags="alert alert-warning", fail_silently=True)
    print("WARNING: {}".format(msg))

def add_messages(request, context):
    """
        Inspect the specified context and add key messages to
        alert the user to key information found.
    """
    assert context is not None
    as_at = context.get('most_recent_date', None)
    sector = context.get('sector', None)
    if as_at:
        info(request, 'Prices current as at {}'.format(as_at))
    if sector:
        info(request, "Only stocks from {} are shown".format(sector))
