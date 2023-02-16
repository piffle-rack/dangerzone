import os
import re
import subprocess
import time
from pathlib import Path
from typing import List

import pytest
from _pytest.fixtures import FixtureRequest

from dangerzone.document import SAFE_EXTENSION
from dangerzone.isolation_provider.container import CONTAINER_LOG_EXT

from .test_cli import TestCli

test_docs_repo_dir = Path(__file__).parent / "test_docs_large"
test_docs_dir = test_docs_repo_dir / "all_documents"
TEST_DOCS_REPO = "git@github.com:freedomofpress/dangerzone-test-set.git"
FORMATS_REGEX = (
    r".*\.(pdf|docx|doc|xlsx|xls|pptx|ppt|odt|ods|odp|odg|jpg|jpeg|gif|png)$"
)


def ensure_test_data_exists() -> None:
    if len(os.listdir(test_docs_repo_dir)) == 0:
        print("Test data repository it empty. Skipping large tests.")
        exit(1)


def get_test_docs(min_size: int, max_size: int) -> List[Path]:
    ensure_test_data_exists()
    return sorted(
        [
            doc
            for doc in test_docs_dir.rglob("*")
            if doc.is_file()
            and min_size < doc.stat().st_size < max_size
            and not (doc.name.endswith(SAFE_EXTENSION))
            and re.match(FORMATS_REGEX, doc.name)
        ]
    )


docs_10K = get_test_docs(min_size=0, max_size=10 * 2**10)
docs_100K = get_test_docs(min_size=10 * 2**10, max_size=100 * 2**10)
docs_10M = get_test_docs(min_size=100 * 2**10, max_size=10 * 2**20)
docs_100M = get_test_docs(min_size=10 * 2**20, max_size=100 * 2**20)

# Pytest parameter decorators
for_each_10K_doc = pytest.mark.parametrize(
    "doc", docs_10K, ids=[str(doc.name) for doc in docs_10K]
)
for_each_100K_doc = pytest.mark.parametrize(
    "doc", docs_100K, ids=[str(doc.name) for doc in docs_100K]
)
for_each_10M_doc = pytest.mark.parametrize(
    "doc", docs_10M, ids=[str(doc.name) for doc in docs_10M]
)
for_each_100M_doc = pytest.mark.parametrize(
    "doc", docs_100M, ids=[str(doc.name) for doc in docs_100M]
)


@pytest.fixture
def training(request: FixtureRequest) -> bool:
    if request.config.getoption("--train"):
        return True
    else:
        return False


class TestLargeSet(TestCli):
    def expected_container_output(self, input_file: Path) -> str:
        # obtains the expected .log file
        output_log_path = f"{input_file}.log"
        with open(output_log_path, "r") as f:
            return f.read()

    def expected_success(self, input_file: Path) -> bool:
        # obtains the expected result
        expected_result_path = f"{input_file}.{CONTAINER_LOG_EXT}"
        with open(expected_result_path, "r") as f:
            last_line = f.readlines()[-1]  # result is in the last line
            if "FAILURE" in last_line:
                return False
            elif "SUCCESS" in last_line:
                return True
            else:
                raise ValueError(
                    f"Container log file ({expected_result_path}) does not contain the result"
                )

    def run_doc_test(self, doc: Path, tmp_path: Path) -> None:
        output_file_path = str(tmp_path / "output.pdf")
        result = self.run_cli(
            ["--output-filename", output_file_path, "--ocr-lang", "eng", str(doc)]
        )
        success = self.expected_success(doc)
        if os.path.exists(output_file_path):
            assert success, "Document was expected to fail but it didn't!"
            result.assert_success()
        else:
            assert not success, "Document was expected to succeed but it didn't!"
            result.assert_failure()

    def train_doc_test(self, doc: Path, tmp_path: Path) -> None:
        container_log_path = f"{doc}.{CONTAINER_LOG_EXT}"
        if Path(container_log_path).exists():
            os.remove(f"{doc}.{CONTAINER_LOG_EXT}")
        output_file_path = str(tmp_path / "output.pdf")
        self.run_cli(
            ["--output-filename", output_file_path, "--ocr-lang", "eng", str(doc)],
            env={"DZ_LOG_CONTAINER": "yes"},
        )

    def test_metadata(  # type: ignore[no-untyped-def]
        self, record_testsuite_property
    ) -> None:
        # Add metadata about the test run to the test suite
        record_testsuite_property(
            "GIT_REF",
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode()[:-1],
        )
        assert True

    @for_each_10K_doc
    def test_10K_docs(self, doc: Path, tmp_path: Path, training: bool) -> None:
        if not training:
            self.run_doc_test(doc, tmp_path)
        else:
            self.train_doc_test(doc, tmp_path)

    @for_each_100K_doc
    def test_100K_docs(self, doc: Path, tmp_path: Path, training: bool) -> None:
        if not training:
            self.run_doc_test(doc, tmp_path)
        else:
            self.train_doc_test(doc, tmp_path)

    @for_each_10M_doc
    def test_10M_docs(self, doc: Path, tmp_path: Path, training: bool) -> None:
        if not training:
            self.run_doc_test(doc, tmp_path)
        else:
            self.train_doc_test(doc, tmp_path)

    @for_each_100M_doc
    def test_100M_docs(self, doc: Path, tmp_path: Path, training: bool) -> None:
        if not training:
            self.run_doc_test(doc, tmp_path)
        else:
            self.train_doc_test(doc, tmp_path)
