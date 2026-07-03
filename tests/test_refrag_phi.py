from scripts.refrag_phi import load_phi_matrix, project, project_batch
import json


def test_phi_load_and_project(tmp_path):
    phi = [
        [1.0, 0.0],
        [0.0, 2.0],
        [-1.0, 1.0],
    ]  # shape (3,2)
    p = tmp_path / "phi.json"
    p.write_text(json.dumps(phi))

    m = load_phi_matrix(str(p))
    assert len(m) == 3 and len(m[0]) == 2
    y = project([2.0, 3.0, 4.0], m)  # [2*1 + 3*0 + 4*(-1), 2*0 + 3*2 + 4*1] = [-2, 10]
    assert y == [-2.0, 10.0]

    ys = project_batch([[1, 0, 0], [0, 1, 0], [0, 0, 1]], m)
    assert ys[0] == [1.0, 0.0]
    assert ys[1] == [0.0, 2.0]
    assert ys[2] == [-1.0, 1.0]
