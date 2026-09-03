# Security and Privacy

Do not commit credentials, private URLs, personal contact information, absolute local paths, raw recordings, photographs, spreadsheets, or unreviewed experiment exports.

Before publishing a change:

1. Review the complete staged diff.
2. Confirm that no secret, token, password, private link, e-mail address, or user-specific path is present.
3. Keep raw measurements under `data/`; this directory is ignored except for its documentation.
4. Publish only curated synthetic examples or results that have been checked for embedded metadata.
5. If sensitive material is committed, revoke exposed credentials immediately and remove the material from the full Git history. Deleting it only in a later commit is not sufficient.

Report security or privacy concerns privately to the repository owner through GitHub rather than opening a public issue containing sensitive details.
