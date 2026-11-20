# Labeling rules

Each item in the audit is a pair: a source issue or pull request that mentions
a target issue in another project.

## Labels

**cause**: One item describes a problem in one project that is caused by a bug,
a change, or a release in the other project. Signs: "fails since version X",
"this is a bug in <other project>", a traceback with frames from the other
project, a pull request that closes the other item.

**related**: The items cover the same topic or the same area, but the text does
not show a cause. Examples: both discuss one feature, or one asks a question
about the other.

**noise**: The mention is incidental. Examples: a long list of links, release
notes, a bot comment, a mass mention.

**unclear**: You cannot decide after about two minutes. Do not guess.

## Rules

1. Judge each pair only from the text of the two items and their titles.
2. The direction is not recorded. A cause may go from upstream to downstream, or
   the other way.
3. If you are not sure between "cause" and "related", choose "related".
4. Do not look at the summary counts before the sample is done.
5. Do not change these rules during a run. If you must change them, write the
   change and the date here, and say which labels it affects.