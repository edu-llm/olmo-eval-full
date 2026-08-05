"""Running olmo-eval as a job on the eduLLM platform.

The platform starts a container, hands it an output prefix in S3 and execs a
command. Everything in this subpackage is what runs on the far side of that: it
reads the platform's environment, evaluates one checkpoint, and puts the results
where the platform will look for them.

It lives inside the installed package rather than beside it because the platform
has no way to run arbitrary repository files. A job reaches this code by pip
installing the project and invoking a module, so anything not packaged is
unreachable at run time.
"""
