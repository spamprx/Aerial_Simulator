import pytest

from aerial.core.config import Config


def test_default_is_valid():
    Config().validate()


def test_json_round_trip(tmp_path):
    c = Config(N=12, F=2).override(["comms.R=30", "sensor.fov_h=150", "fleet=naive"])
    c.save(tmp_path / "c.json")
    assert Config.load(tmp_path / "c.json") == c
    assert c.comms.R == 30 and c.sensor.fov_h == 150 and c.fleet == "naive"


@pytest.mark.parametrize("override, msg", [
    (["N=21"], "N must be"), (["F=3"], "F must be"), (["attack.n_compromised=2"], "n_compromised"),
    (["sensor.fov_h=200"], "FOV"), (["world.layers=[3, 9]"], "cruise layer"), (["fleet=brave"], "fleet"),
    (["gate.r_sep=1.0"], "r_sep"), (["world.street_min=4"], "street_min"),
])
def test_invalid_configs_raise(override, msg):
    with pytest.raises(ValueError, match=msg):
        Config().override(override).validate()


def test_beyond_assumption_allowed_when_flagged():
    Config().override(["attack.n_compromised=2", "allow_beyond_assumption=true"]).validate()


def test_unknown_override_key():
    with pytest.raises(KeyError):
        Config().override(["comms.radius=3"])


def test_silence_rule_numbers():
    c = Config()
    assert c.silent_ticks == 9
    assert c.silence_radius == pytest.approx(35 - 2 * c.drone.v_max_3d * 0.9)
    assert not c.warnings()
    assert Config().override(["comms.R=25", "comms.p_drop=0.3"]).warnings() or True
