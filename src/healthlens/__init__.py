"""HealthLens — private, offline analysis of an Apple Health export.

Pipeline:
    export.zip --(parser)--> Parquet cache --(metrics)--> daily/sleep/workout tables
               --(analysis)--> insights --(report / dashboard)--> offline HTML

Nothing in this package makes network requests.
"""

__version__ = "0.1.0"
