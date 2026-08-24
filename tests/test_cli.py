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


def test_run_name_distinguishes_ablation_artifacts() -> None:
    args = build_parser().parse_args(["run-baseline", "--run-name", "tfidf_lr_title_only"])
    assert args.run_name == "tfidf_lr_title_only"
