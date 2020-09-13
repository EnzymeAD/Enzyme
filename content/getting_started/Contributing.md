---
title: "How to Contribute"
date: 2019-11-29T15:26:15Z
draft: false
weight: 15
---

Everyone is welcome to contribute to MLIR. There are several ways of getting
involved and contributing including reporting bugs, improving documentation and
tutorials.

## Community Guidelines

Please be mindful of the [LLVM Code of Conduct](https://llvm.org/docs/CodeOfConduct.html),
which pledges to foster an open and welcoming environment.

### Contributing code

We don't accept pull-request on GitHub, instead we use
[Phabricator](https://llvm.org/docs/Phabricator.html).
At the moment you need to also join this [group](https://reviews.llvm.org/project/members/78/)
to enable build and test of your Phabricator revisions.

Once a patch is approved on Phabricator and pass continuous integration checks,
it can be pushed directly to the master branch of the repository.

#### Using Arcanist

Use [Arcanist](https://llvm.org/docs/Phabricator.html#requesting-a-review-via-the-command-line)
to send your patches for review. This triggers the continuous build system and
preserves the authorship information in case somebody else commits the patch on
your behalf.

You can also setup a check as a git hook to prevent accidentally pushing
unformatted commits with:
`ln -sf ../../llvm/utils/git/pre-push.py .git/hooks/pre-push`

#### First-time contributors

LLVM follows a [policy](https://llvm.org/docs/DeveloperPolicy.html#obtaining-commit-access)
of granting established contributors direct commit access to the repository.
If you make your first contributions to the LLVM infrastructure, you are unlikely
to have commit access. In this case, ask the reviewers to commit the change
for you after it has been accepted. Once you have demonstrated the ability to
write high-quality patches, follow the
[policy](https://llvm.org/docs/DeveloperPolicy.html#obtaining-commit-access) to
obtain commit access for yourself.

*Note:* if you haven't used Arcanist to send your patch for review, committers
don't have access to your preferred identity for commit messages. Make sure
to communicate it to them through available channels or use the git sign-off
functionality to make your identity visible in the commit message.

#### Commit messages

Follow the git conventions for writing a commit message, in particular the
first line is the short title of the commit. The title should be followed by an
empty line and a longer description. Prefer describing *why* the change is
implemented rather than what it does. The latter can be inferred from the code.
This [post](https://chris.beams.io/posts/git-commit/) give examples and more
details.

### Issue tracking

To report a bug, use the [MLIR product on the LLVM bug
tracker](https://bugs.llvm.org/enter_bug.cgi?product=MLIR), try to pick a
suitable component for the bug, or leave it in the default.

If you want to contribute, start working through the MLIR codebase, navigate to
[the "beginner" issues](https://bugs.llvm.org/buglist.cgi?keywords=beginner%2C%20&keywords_type=allwords&list_id=176893&product=MLIR&query_format=advanced&resolution=---)
and start looking through interesting issues. If you decide to start on an
issue, assign it to yourself and leave a comment so that other people know that
you're working on it. If you want to help out, but not alone, use the issue
comment thread to coordinate.

### Contribution guidelines and standards

*   Read the [developer guide](DeveloperGuide.md "here").
*   Ensure that you use the correct license. Examples are provided below.
*   Include tests when you contribute new features, as they help to a)
    prove that your code works correctly, and b) guard against future breaking
    changes to lower the maintenance cost.
*   Bug fixes also generally require tests, because the presence of bugs
    usually indicates insufficient test coverage.

#### License

Include a license at the top of new files.

* [C/C++ license example](https://github.com/llvm/llvm-project/blob/master/mlir/examples/toy/Ch1/toyc.cpp)
