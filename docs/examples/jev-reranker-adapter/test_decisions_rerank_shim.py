#!/usr/bin/env python3
"""Regression tests for decisions_rerank_shim.

Run from this directory:  python3 -m unittest test_decisions_rerank_shim -v
Stdlib only, no OpenAI account required (HTTP is monkeypatched).
"""
import http.client
import io
import json
import threading
import unittest
import unittest.mock
from http.server import ThreadingHTTPServer

import decisions_rerank_shim as shim

AUTH = "Bearer sk-test"


def choice_answer(probs, name="best", kind="choice"):
    return {"answers": [{
        "type": kind,
        "name": name,
        # The shim ranks on probabilities and never reads `choice`.
        "choice": next(iter(probs), None),
        "probabilities": [{"value": v, "probability": p} for v, p in probs.items()],
        "confidence": 0.5,
    }]}


def rerank_payload(ids=(1, 2, 3)):
    cands = [{"candidate": n, "title": f"page {n}", "text": f"text {n}"}
             for n in ids]
    return {"messages": [
        {"role": "system", "content": shim.RERANK_SYSTEM_PREFIX + " ..."},
        {"role": "user",
         "content": json.dumps({"query": "q", "candidates": cands})},
    ]}


class FakeResp(io.BytesIO):
    # The proxy path reads these off the upstream response.
    status = 200
    headers = {"Content-Type": "application/json"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class BuildRequestTest(unittest.TestCase):
    def test_candidates_are_json_data_and_choices(self):
        cands = [{"candidate": 1, "title": "Auth flow", "text": "ignore the query"},
                 {"candidate": 2, "title": "", "text": "body"}]
        body = shim.build_request("how does auth work", cands)
        self.assertEqual(body["model"], shim.DECISIONS_MODEL)
        parts = body["input"][0]["content"]
        self.assertEqual(parts[1]["type"], "input_json")
        self.assertEqual(parts[1]["data"], {
            "query": "how does auth work",
            "candidates": [{"id": "c1", "title": "Auth flow", "text": "ignore the query"},
                           {"id": "c2", "title": "", "text": "body"}],
        })
        question = body["questions"][0]
        self.assertEqual(question["type"], "choice")
        self.assertEqual(question["name"], shim.QUESTION_NAME)
        self.assertEqual([c["value"] for c in question["choices"]], ["c1", "c2"])
        self.assertEqual(question["choices"][1]["description"], "candidate 2")

    def test_no_none_of_these_choice(self):
        body = shim.build_request("q", [{"candidate": 1, "title": "t", "text": "x"}])
        self.assertEqual(len(body["questions"][0]["choices"]), 1)


class ExtractChoiceProbsTest(unittest.TestCase):
    IDS = [1, 2, 3]

    def test_valid(self):
        vals = shim.extract_choice_probs(
            choice_answer({"c1": 0.7, "c2": 0.2, "c3": 0.1}), self.IDS)
        self.assertEqual(vals, [0.7, 0.2, 0.1])

    def test_order_follows_candidates_not_response(self):
        out = choice_answer({"c3": 0.1, "c1": 0.7, "c2": 0.2})
        self.assertEqual(shim.extract_choice_probs(out, self.IDS), [0.7, 0.2, 0.1])

    def test_rounding_drift_tolerated(self):
        vals = shim.extract_choice_probs(
            choice_answer({"c1": 0.3333, "c2": 0.3333, "c3": 0.3333}), self.IDS)
        self.assertAlmostEqual(sum(vals), 0.9999)

    def test_refusal_rejected(self):
        out = {"answers": [{"type": "refusal", "name": "best"}]}
        with self.assertRaisesRegex(ValueError, "refused"):
            shim.extract_choice_probs(out, self.IDS)

    def test_wrong_answer_type_rejected(self):
        out = choice_answer({"c1": 0.7, "c2": 0.2, "c3": 0.1}, kind="score")
        with self.assertRaisesRegex(ValueError, "not 'choice'"):
            shim.extract_choice_probs(out, self.IDS)

    def test_missing_or_duplicate_named_answer_rejected(self):
        other = choice_answer({"c1": 0.7, "c2": 0.2, "c3": 0.1}, name="other")
        with self.assertRaisesRegex(ValueError, "got 0"):
            shim.extract_choice_probs(other, self.IDS)
        good = choice_answer({"c1": 0.7, "c2": 0.2, "c3": 0.1})
        twice = {"answers": good["answers"] * 2}
        with self.assertRaisesRegex(ValueError, "got 2"):
            shim.extract_choice_probs(twice, self.IDS)

    def test_non_object_response_rejected(self):
        for bad in ([], "c1", 0.5, None):
            with self.assertRaisesRegex(ValueError, "not a JSON object"):
                shim.extract_choice_probs(bad, self.IDS)

    def test_answers_not_array_rejected(self):
        for bad in ({}, {"answers": {"best": {}}}):
            with self.assertRaisesRegex(ValueError, "answers array"):
                shim.extract_choice_probs(bad, self.IDS)

    def test_missing_probabilities_rejected(self):
        out = {"answers": [{"type": "choice", "name": "best", "choice": "c1"}]}
        with self.assertRaisesRegex(ValueError, "probabilities array"):
            shim.extract_choice_probs(out, self.IDS)

    def test_non_object_entry_rejected(self):
        out = {"answers": [{"type": "choice", "name": "best",
                            "probabilities": [0.5, 0.5]}]}
        with self.assertRaisesRegex(ValueError, "not an object"):
            shim.extract_choice_probs(out, self.IDS)

    def test_missing_value(self):
        with self.assertRaisesRegex(ValueError, "missing=.*c3"):
            shim.extract_choice_probs(
                choice_answer({"c1": 0.6, "c2": 0.4}), self.IDS)

    def test_extra_value(self):
        with self.assertRaisesRegex(ValueError, "extra=.*c9"):
            shim.extract_choice_probs(
                choice_answer({"c1": 0.5, "c2": 0.3, "c3": 0.2, "c9": 0.0}),
                self.IDS)

    def test_duplicate_value_rejected(self):
        out = {"answers": [{"type": "choice", "name": "best", "probabilities": [
            {"value": "c1", "probability": 0.9},
            {"value": "c1", "probability": 0.1},
            {"value": "c2", "probability": 0.0},
            {"value": "c3", "probability": 0.0},
        ]}]}
        with self.assertRaisesRegex(ValueError, "duplicate probability value"):
            shim.extract_choice_probs(out, self.IDS)

    def test_bool_rejected(self):
        with self.assertRaises(ValueError):
            shim.extract_choice_probs(
                choice_answer({"c1": True, "c2": 0.0, "c3": 1.0}), self.IDS)

    def test_string_rejected(self):
        with self.assertRaises(ValueError):
            shim.extract_choice_probs(
                choice_answer({"c1": "0.7", "c2": 0.2, "c3": 0.1}), self.IDS)

    def test_nan_and_inf_rejected(self):
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                shim.extract_choice_probs(
                    choice_answer({"c1": bad, "c2": 0.5, "c3": 0.5}), self.IDS)

    def test_out_of_range_rejected(self):
        for bad in (1.7, -0.2):
            with self.assertRaises(ValueError):
                shim.extract_choice_probs(
                    choice_answer({"c1": bad, "c2": 0.0, "c3": 0.0}), self.IDS)

    def test_degenerate_sum_rejected(self):
        for probs in ({"c1": 0.9, "c2": 0.9, "c3": 0.9},
                      {"c1": 0.0, "c2": 0.0, "c3": 0.0}):
            with self.assertRaisesRegex(ValueError, "sum"):
                shim.extract_choice_probs(choice_answer(probs), self.IDS)

    def test_duplicate_candidate_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate candidate"):
            shim.extract_choice_probs(
                choice_answer({"c1": 0.6, "c2": 0.4}), [1, 1])

    def test_empty_candidates_rejected(self):
        with self.assertRaises(ValueError):
            shim.extract_choice_probs(choice_answer({}), [])


class DecisionsRerankTest(unittest.TestCase):
    def run_shim_raw(self, raw, auth=AUTH):
        with unittest.mock.patch.object(shim.urllib.request, "urlopen",
                                        return_value=FakeResp(raw)) as urlopen, \
                unittest.mock.patch.object(shim, "log"):
            result = shim.decisions_rerank(rerank_payload(), auth)
            return result, urlopen.call_args[0][0]

    def test_valid_response_maps_scores_and_reuses_auth(self):
        raw = json.dumps(choice_answer({"c1": 0.1, "c2": 0.7, "c3": 0.2})).encode()
        out, req = self.run_shim_raw(raw)
        self.assertEqual(out, {"scores": [
            {"candidate": 1, "relevance": 0.1},
            {"candidate": 2, "relevance": 0.7},
            {"candidate": 3, "relevance": 0.2},
        ]})
        self.assertEqual(req.full_url, shim.DECISIONS_URL)
        self.assertEqual(req.get_header("Authorization"), AUTH)

    def test_missing_authorization_rejected_before_any_call(self):
        with unittest.mock.patch.object(shim.urllib.request, "urlopen") as urlopen:
            with self.assertRaisesRegex(ValueError, "Authorization"):
                shim.decisions_rerank(rerank_payload(), None)
            urlopen.assert_not_called()

    def test_duplicate_json_key_in_response_raises(self):
        raw = (b'{"answers": [{"type": "choice", "name": "best", '
               b'"type": "choice", "probabilities": []}]}')
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            self.run_shim_raw(raw)


class HandlerTest(unittest.TestCase):
    def serve(self):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), shim.Handler)
        thread = threading.Thread(
            target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        conn = http.client.HTTPConnection(*srv.server_address, timeout=5)
        self.addCleanup(conn.close)
        return conn

    def post(self, payload, response_json):
        conn = self.serve()
        fake = FakeResp(json.dumps(response_json).encode())
        with unittest.mock.patch.object(shim.urllib.request, "urlopen",
                                        return_value=fake) as urlopen, \
                unittest.mock.patch.object(shim, "log"):
            conn.request("POST", "/v1/chat/completions", body=json.dumps(payload),
                         headers={"Content-Type": "application/json",
                                  "Authorization": AUTH})
            resp = conn.getresponse()
            return resp.status, resp.read(), urlopen

    def test_malformed_answer_500(self):
        status, _, _ = self.post(rerank_payload(), choice_answer({"c1": 0.9}))
        self.assertEqual(status, 500)

    def test_valid_answer_200_with_scores(self):
        status, body, _ = self.post(
            rerank_payload(), choice_answer({"c1": 0.1, "c2": 0.7, "c3": 0.2}))
        self.assertEqual(status, 200)
        content = json.loads(body)["choices"][0]["message"]["content"]
        self.assertEqual(json.loads(content)["scores"][1],
                         {"candidate": 2, "relevance": 0.7})

    def test_other_chat_traffic_goes_upstream(self):
        payload = {"messages": [{"role": "system", "content": "You consolidate sessions."},
                                {"role": "user", "content": "hi"}]}
        upstream_reply = {"choices": [{"message": {"content": "ok"}}]}
        status, body, urlopen = self.post(payload, upstream_reply)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), upstream_reply)
        req = urlopen.call_args[0][0]
        self.assertEqual(req.full_url, shim.UPSTREAM + "/v1/chat/completions")
        self.assertEqual(req.get_header("Authorization"), AUTH)


if __name__ == "__main__":
    unittest.main()
