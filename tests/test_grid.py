"""Контракт сетки ячеек — то, что ломается молча и дорого.

Смена шага широты с 1° на 0.75° переписала id ячейки. Здесь закреплено ровно то,
на что опираются кэш GFS, датасет, артефакт и фронт: формат id, отсутствие
пересечений с прежним форматом, полуоткрытые границы и якорь на узле сетки GFS.
"""

import pytest

from paraglideml.grid import (
    LAT_STEP,
    LON_STEP,
    cell_anchor,
    cell_bounds,
    cell_id,
    cell_ring,
    cells_bbox,
    cells_in_bbox,
    contains,
)


def test_id_format_keeps_two_decimals():
    # Два знака обязательны: "45_11" в прежнем контракте означал квадрат 45-46°,
    # и совпадение id разных сеток дало бы тихий сдвиг вместо ошибки.
    assert cell_id(45.9, 11.3) == "45.75_11"
    assert cell_id(45.0, 11.0) == "45.00_11"
    assert cell_id(45.0, 11.0) != "45_11"


def test_negative_longitude_survives_the_split():
    cid = cell_id(36.4, -5.46)
    assert cid == "36.00_-6"
    assert cell_anchor(cid) == (36.0, -6.0)


def test_cell_id_is_idempotent_on_its_own_anchor():
    for lat, lon in [(36.0, -6), (45.75, 11), (52.5, 16), (38.25, -3)]:
        assert cell_id(lat, lon) == cell_id(*cell_anchor(cell_id(lat, lon)))


def test_anchors_land_on_the_gfs_quarter_degree_grid():
    # Ради этого шаг 0.75, а не геометрический оптимум 0.698: иначе якорь ячейки
    # уплывает между узлами GFS и точка выборки перестаёт быть определённой.
    for cid in cells_in_bbox(-8, 36, 30, 55):
        lat0, lon0 = cell_anchor(cid)
        assert abs((lat0 / 0.25) - round(lat0 / 0.25)) < 1e-9
        assert abs((lon0 / 0.25) - round(lon0 / 0.25)) < 1e-9


def test_bounds_are_half_open_so_a_border_point_belongs_to_one_cell():
    cid = cell_id(46.0, 13.0)
    lat0, lon0, lat1, lon1 = cell_bounds(cid)
    assert contains(cid, lat0, lon0)
    assert not contains(cid, lat1, lon0)  # верхняя грань — уже соседняя ячейка
    assert not contains(cid, lat0, lon1)
    assert cell_id(lat1, lon0) != cid


def test_cell_size_matches_the_declared_steps():
    lat0, lon0, lat1, lon1 = cell_bounds("45.75_11")
    assert lat1 - lat0 == pytest.approx(LAT_STEP)
    assert lon1 - lon0 == pytest.approx(LON_STEP)


def test_ring_is_closed_and_matches_bounds():
    ring = cell_ring("45.75_11")
    assert ring[0] == ring[-1]
    assert len(ring) == 5
    lat0, lon0, lat1, lon1 = cell_bounds("45.75_11")
    assert [lon0, lat0] in ring and [lon1, lat1] in ring


def test_cells_bbox_covers_every_cell_upper_edge_included():
    cells = ["36.00_-6", "55.50_24"]
    lon_min, lat_min, lon_max, lat_max = cells_bbox(cells)
    assert (lon_min, lat_min) == (-6.0, 36.0)
    assert (lon_max, lat_max) == (25.0, 55.5 + LAT_STEP)
    # bbox должен вернуть сам себя: иначе крайняя ячейка выпадает из извлечения GFS
    assert set(cells) <= set(cells_in_bbox(lon_min, lat_min, lon_max, lat_max))


def test_malformed_id_raises_instead_of_guessing():
    with pytest.raises(ValueError):
        cell_anchor("не_ячейка")


@pytest.mark.parametrize("legacy", ["45_11", "36_-6", "45.0_11", "45.750_11"])
def test_legacy_id_raises_rather_than_shifting_half_a_degree(legacy):
    # float("45") разбирается молча, и "45_11" дал бы ячейку 45.00-45.75 вместо
    # прежнего квадрата 45-46°. Ради этого у широты ровно два знака — и они проверяются.
    with pytest.raises(ValueError):
        cell_anchor(legacy)
