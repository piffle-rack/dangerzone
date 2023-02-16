from pytest import Parser


def pytest_addoption(parser: Parser) -> None:
    """Adds the --train option to the tests. This is"""
    parser.addoption(
        "--train", action="store_true", help="Enable training of large document set"
    )
