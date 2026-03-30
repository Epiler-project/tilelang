from tilelang.utils.target import determine_target


def test_determine_target_normalizes_riscv_alias():
    assert determine_target("riscv") == "linalg_riscv"


def test_determine_target_accepts_linalg_riscv():
    assert determine_target("linalg_riscv") == "linalg_riscv"
