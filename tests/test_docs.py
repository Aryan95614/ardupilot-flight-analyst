"""DocRecommender tests."""

import pytest

from flight_analyst.docs import (
    FAILURE_DOCS,
    WIKI_URLS,
    DocRecommender,
    _adjust_url_for_vehicle,
)


@pytest.fixture
def recommender() -> DocRecommender:
    return DocRecommender()


class TestWikiURLs:
    def test_entry_count(self):
        assert len(WIKI_URLS) == 23

    def test_all_urls_start_with_https(self):
        for key, url in WIKI_URLS.items():
            assert url.startswith("https://"), f"{key} URL invalid"

    def test_all_urls_contain_copter_base(self):
        for key, url in WIKI_URLS.items():
            assert "/copter/docs/" in url, f"{key} missing /copter/docs/"


class TestFailureDocs:
    def test_covers_all_failure_classes(self):
        assert len(FAILURE_DOCS) == 15

    def test_every_wiki_key_exists(self):
        for fc, entries in FAILURE_DOCS.items():
            for title, wiki_key, note in entries:
                assert wiki_key in WIKI_URLS, (
                    f"Failure class '{fc}' references unknown wiki key "
                    f"'{wiki_key}'"
                )


class TestURLAdjustment:
    def test_copter_unchanged(self):
        url = "https://ardupilot.org/copter/docs/tuning.html"
        assert _adjust_url_for_vehicle(url, "Copter") == url

    def test_plane_adjustment(self):
        url = "https://ardupilot.org/copter/docs/tuning.html"
        result = _adjust_url_for_vehicle(url, "Plane")
        assert "/plane/docs/" in result
        assert "/copter/" not in result

    def test_rover_adjustment(self):
        url = "https://ardupilot.org/copter/docs/tuning.html"
        result = _adjust_url_for_vehicle(url, "Rover")
        assert "/rover/docs/" in result

    def test_sub_adjustment(self):
        url = "https://ardupilot.org/copter/docs/tuning.html"
        result = _adjust_url_for_vehicle(url, "Sub")
        assert "/sub/docs/" in result

    def test_unknown_vehicle_defaults_to_copter(self):
        url = "https://ardupilot.org/copter/docs/tuning.html"
        assert _adjust_url_for_vehicle(url, "Blimp") == url


class TestRecommend:
    def test_high_vibration(self, recommender):
        recs = recommender.recommend("high_vibration")
        assert len(recs) == 3
        assert recs[0]["title"] == "Measuring Vibration"

    def test_unknown_failure_class_returns_empty(self, recommender):
        assert recommender.recommend("nonexistent_class") == []

    def test_result_dict_keys(self, recommender):
        recs = recommender.recommend("compass_error")
        for r in recs:
            assert set(r.keys()) == {"title", "url", "relevance_note"}

    def test_vehicle_type_rewrites_url(self, recommender):
        recs = recommender.recommend("crash", vehicle_type="Plane")
        for r in recs:
            assert "/plane/docs/" in r["url"]
            assert "/copter/" not in r["url"]

    def test_every_failure_class_has_recommendations(self, recommender):
        for fc in FAILURE_DOCS:
            recs = recommender.recommend(fc)
            assert len(recs) > 0, f"No recommendations for '{fc}'"

    def test_default_vehicle_is_copter(self, recommender):
        recs = recommender.recommend("ekf_failure")
        for r in recs:
            assert "/copter/docs/" in r["url"]


class TestGetAllFailureClasses:
    def test_returns_sorted_list(self, recommender):
        classes = recommender.get_all_failure_classes()
        assert classes == sorted(classes)

    def test_count_matches_failure_docs(self, recommender):
        assert len(recommender.get_all_failure_classes()) == len(FAILURE_DOCS)

    def test_contains_known_class(self, recommender):
        assert "high_vibration" in recommender.get_all_failure_classes()
