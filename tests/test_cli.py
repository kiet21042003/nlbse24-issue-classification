from nlbse24.cli import build_parser


def test_official_test_requires_explicit_confirmation() -> None:
    args = build_parser().parse_args(["run-baseline", "--protocol", "official"])
    assert args.confirm_official_test is False


def test_repository_filter_can_be_repeated() -> None:
    args = build_parser().parse_args(
        [
            "run-baseline",
            "--repository",
            "facebook/react",
            "--repository",
            "opencv/opencv",
        ]
    )
    assert args.repository == ["facebook/react", "opencv/opencv"]
