from relay_leader_daily_v236 import parser


def test_daily_parser_requires_all_forward_registries():
    args = parser().parse_args([
        "--review", "review.json", "--raw-limit-up", "limit.csv",
        "--announcements", "ann.csv", "--announcement-manifest", "manifest.json",
        "--same-height-registry", "v226", "--role-registry", "v236",
        "--cross-theme-registry", "v241",
        "--path-expectation-registry", "v244",
        "--output", "out",
    ])
    assert args.same_height_registry == "v226"
    assert args.role_registry == "v236"
    assert args.cross_theme_registry == "v241"
    assert args.path_expectation_registry == "v244"
