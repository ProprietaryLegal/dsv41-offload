"""Scorer behaviour on hand-written replies (synthetic strings only; no model output)."""
from __future__ import annotations

import pytest

from offload.lsuite.scoring import (
    CitationKey, canonical_figure, find_dates, first_value, normalize, score_citation, score_figure, score_needle,
    score_quote, score_refusal)

SENTENCE = "The family court\nfound that the husband\u2019s testimony was not credible on this point."


def test_normalize_collapses_layout_and_typography() -> None:
    assert normalize("  a\n\nb\u00a0c \u201cd\u201d \u2014 e\u2019s ") == "a b c \"d\" - e's"


@pytest.mark.parametrize(("reply", "passed", "reason"), [
    ("The family court found that the husband's testimony was not credible on this point.", True, "verbatim"),
    ('"The family court found that the husband\u2019s testimony was not credible on this point."', True, "verbatim"),
    ("Quote: The family court found that the husband's testimony\nwas not credible on this point.", True, "verbatim"),
    ("The family court found that the husband's testimony was not credible on this point", False, "not-verbatim"),
    ("the family court found that the husband's testimony was not credible on this point.", False, "not-verbatim"),
    ("The family court found the husband's testimony was not credible on this point.", False, "not-verbatim"),
    ("NOT IN RECORD", False, "refused"),
    # S101 FIX 2 repros: markdown emphasis is typography; the words must still be verbatim
    ("The *family court* found that the husband's testimony was not credible on this point.", True, "verbatim"),
    ("The **family court** found that the husband's testimony was _not_ credible on this point.", True, "verbatim"),
    ("The *family court* found that the husband's testimony was credible on this point.", False, "not-verbatim"),
])
def test_quote(reply: str, passed: bool, reason: str) -> None:
    score = score_quote(reply, SENTENCE)
    assert (score.passed, score.reason) == (passed, reason)


def test_quote_refusal_wins_over_containment() -> None:
    reply = "The family court found that the husband's testimony was not credible on this point. Not in the record."
    assert score_quote(reply, SENTENCE).reason == "refused"


@pytest.mark.parametrize(("kind", "raw", "value"), [
    ("money", "$16,000", "16000"),
    ("money", "$1,234.50", "1234.5"),
    ("money", "$ 250", "250"),
    ("percent", "35%", "35"),
    ("percent", "12.5 percent", "12.5"),
    ("date", "June 3, 2019", "2019-06-03"),
    ("date", "Sept. 30, 2021", "2021-09-30"),
])
def test_canonical_figure(kind: str, raw: str, value: str) -> None:
    assert canonical_figure(kind, raw) == value  # type: ignore[arg-type]


def test_canonical_figure_refuses_unparseable_span() -> None:
    with pytest.raises(ValueError, match="does not parse"):
        canonical_figure("date", "the thirtieth of June")


@pytest.mark.parametrize(("kind", "reply", "value", "passed", "reason"), [
    ("money", "$16,000", "16000", True, "match"),
    ("money", "$16,000.00.", "16000", True, "match"),
    ("money", "16,000", "16000", True, "match"),
    ("money", "$15,000", "16000", False, "wrong-value"),
    ("money", "The court reduced it from $15,000 to $16,000.", "16000", False, "wrong-value"),
    ("money", "sixteen thousand dollars", "16000", False, "no-value"),
    ("money", "NOT IN RECORD", "16000", False, "refused"),
    ("percent", "35%", "35", True, "match"),
    ("percent", "35 percent", "35", True, "match"),
    ("percent", "35.0 %", "35", True, "match"),
    ("percent", "53%", "35", False, "wrong-value"),
    ("date", "June 3, 2019", "2019-06-03", True, "match"),
    ("date", "3 June 2019", "2019-06-03", True, "match"),
    ("date", "2019-06-03", "2019-06-03", True, "match"),
    ("date", "06/03/2019", "2019-06-03", True, "match"),
    ("date", "June 3rd, 2019", "2019-06-03", True, "match"),
    ("date", "June 30, 2019", "2019-06-03", False, "wrong-value"),
    ("date", "in June of 2019", "2019-06-03", False, "no-value"),
])
def test_figure(kind: str, reply: str, value: str, passed: bool, reason: str) -> None:
    score = score_figure(reply, kind, value)  # type: ignore[arg-type]
    assert (score.passed, score.reason) == (passed, reason)


def test_find_dates_orders_by_position_and_skips_impossible() -> None:
    reply = normalize("February 30, 2020 then 2021-01-05 and March 1, 2022")
    assert find_dates(reply) == [(reply.index("2021"), "2021-01-05"), (reply.index("March"), "2022-03-01")]
    assert first_value("date", "nothing here") is None


SC_KEY = CitationKey((("325", "S.C.", "456"), ("481", "S.E.2d", "174")), "1997", "Ct. App.", True)
SC_SUPREME = CitationKey((("345", "S.C.", "88"), ("545", "S.E.2d", "531")), "2001", "", True)
FED_KEY = CitationKey((("550", "U.S.", "544"),), "2007", "", False)
CIR_KEY = CitationKey((("901", "F.3d", "120"),), "2018", "4th Cir.", False)
TENN_KEY = CitationKey((("350", "S.W.3d", "99"),), "2011", "Tenn.", False)


@pytest.mark.parametrize(("key", "reply", "passed", "reason"), [
    (SC_KEY, "Belton v. Belton, 325 S.C. 456, 481 S.E.2d 174 (Ct. App. 1997)", True, "match"),
    (SC_KEY, "Belton v. Belton, 325 S.C.\n456, 460, 481 S.E.2d 174, 176 (Ct. App. 1997).", True, "match"),
    (SC_KEY, "Belton v. Belton, 325 S. C. 456, 481 S.E. 2d 174 (Ct.App. 1997)", True, "match"),
    (SC_KEY, "Belton v. Belton, 481 S.E.2d 174, 325 S.C. 456 (Ct. App. 1997)", False, "order"),
    (SC_KEY, "Belton v. Belton, 325 S.C. 456, 481 S.E.2d 147 (Ct. App. 1997)", False, "missing-reporter"),
    (SC_KEY, "Belton v. Belton, 325 S.C. 456, 481 S.E.2d 174 (1997)", False, "wrong-court"),
    (SC_KEY, "Belton v. Belton, 325 S.C. 456, 481 S.E.2d 174 (Ct. App. 1998)", False, "missing-year"),
    (SC_KEY, "Belton v. Belton, 325 S.C. 456, 481 S.E.2d 174", False, "missing-year"),
    (SC_KEY, "NOT IN RECORD", False, "refused"),
    (SC_SUPREME, "Jenkins v. Jenkins, 345 S.C. 88, 545 S.E.2d 531 (2001)", True, "match"),
    (SC_SUPREME, "Jenkins v. Jenkins, 345 S.C. 88, 545 S.E.2d 531 (Ct. App. 2001)", False, "wrong-court"),
    (SC_SUPREME, "Jenkins v. Jenkins, 345 S.C. 188, 545 S.E.2d 531 (2001)", False, "missing-reporter"),
    (FED_KEY, "Bell Atl. Corp. v. Twombly, 550 U.S. 544, 570 (2007)", True, "match"),
    (FED_KEY, "Bell Atl. Corp. v. Twombly, 1550 U.S. 5440 (2007)", False, "missing-reporter"),
    (CIR_KEY, "Doe v. Roe, 901 F.3d 120, 125 (4th Cir. 2018)", True, "match"),
    (CIR_KEY, "Doe v. Roe, 901 F.3d 120 (2018)", False, "wrong-court"),
    # S101 FIX 3 repros: the parenthetical court must be the key's court, not merely contain it
    (TENN_KEY, "Gonsewski v. Gonsewski, 350 S.W.3d 99, 109 (Tenn. 2011)", True, "match"),
    (TENN_KEY, "Gonsewski v. Gonsewski, 350 S.W.3d 99 (Tenn. Ct. App. 2011)", False, "wrong-court"),
    (SC_SUPREME, "Jenkins v. Jenkins, 345 S.C. 88, 545 S.E.2d 531 (S.C. 2001)", True, "match"),
    (SC_SUPREME, "Jenkins v. Jenkins, 345 S.C. 88, 545 S.E.2d 531 (4th Cir. 2001)", False, "wrong-court"),
    (SC_SUPREME, "Jenkins v. Jenkins, 345 S.C. 88, 545 S.E.2d 531 (N.C. Ct. 2001)", False, "wrong-court"),
    (SC_KEY, "Belton v. Belton, 325 S.C. 456, 481 S.E.2d 174 (S.C. Ct. App. 1997)", True, "match"),
    (CIR_KEY, "Doe v. Roe, 901 F.3d 120 (D.S.C. 2018)", False, "wrong-court"),
])
def test_citation(key: CitationKey, reply: str, passed: bool, reason: str) -> None:
    score = score_citation(reply, key)
    assert (score.passed, score.reason) == (passed, reason)


@pytest.mark.parametrize(("reply", "passed", "reason"), [
    ("2031-DR-47-0839", True, "match"),
    ("The docket number is 2031\u2013DR\u201347\u20130839.", True, "match"),
    ("2031-dr-47-0839", True, "match"),
    ("2031-DR-47-0893", False, "wrong-value"),
    ("NOT IN RECORD", False, "refused"),
    # S101 FIX 1 repros: a near-miss docket must not pass as a substring; spaces for dashes are typography
    ("2031-DR-47-08397", False, "wrong-value"),
    ("12031-DR-47-0839", False, "wrong-value"),
    ("2031-DR-47-08039", False, "wrong-value"),
    ("2031-DR-47-0839 or 2019-DR-40-0001", False, "extra-docket"),
    ("2031 DR 47 0839", True, "match"),
    ("Docket No. 2031DR470839.", True, "match"),
])
def test_needle(reply: str, passed: bool, reason: str) -> None:
    score = score_needle(reply, "2031-DR-47-0839")
    assert (score.passed, score.reason) == (passed, reason)


@pytest.mark.parametrize(("kind", "reply", "passed", "reason"), [
    ("money", "NOT IN RECORD", True, "marker"),
    ("money", "Not in the record.", True, "marker"),
    ("money", "NOT IN RECORD. The opinion does not state any fee amount.", True, "marker"),
    ("money", "NOT IN RECORD, but likely $5,000.", False, "invented-value"),
    ("money", "$5,000", False, "answered"),
    ("date", "NOT IN RECORD", True, "marker"),
    ("date", "NOT IN RECORD (perhaps June 1, 2001)", False, "invented-value"),
    ("date", "June 1, 2001", False, "answered"),
    ("percent", "NOT IN RECORD", True, "marker"),
    ("percent", "NOT IN RECORD; probably 50%", False, "invented-value"),
    ("name", "NOT IN RECORD", True, "marker"),
    ("name", "NOT IN RECORD. The opinion does not name a guardian ad litem.", True, "marker"),
    ("name", "NOT IN RECORD. No Guardian Ad Litem is named.", True, "marker"),
    ("name", "NOT IN RECORD, though Jane Doe may have served.", False, "invented-value"),
    ("name", "NOT IN RECORD, though Jane Q. Doe may have served.", False, "invented-value"),
    ("name", "The guardian was Jane Doe.", False, "answered"),
    # S101 FIX 4 repros: invented values in forms other than $, a full date or a two-word name
    ("money", "NOT IN RECORD. The court awarded 5,000 dollars in fees.", False, "invented-value"),
    ("money", "NOT IN RECORD; fees of five thousand dollars were awarded.", False, "invented-value"),
    ("money", "NOT IN RECORD. The court ordered him to pay two thousand five hundred dollars.", False,
     "invented-value"),
    ("money", "NOT IN RECORD. Fees were USD 5000.", False, "invented-value"),
    ("date", "NOT IN RECORD. They married in June 2001.", False, "invented-value"),
    ("date", "NOT IN RECORD. The parties married in 2001.", False, "invented-value"),
    ("name", "NOT IN RECORD. Ms. Roe served as guardian ad litem.", False, "invented-value"),
    ("name", "NOT IN RECORD. The guardian was Roe.", False, "invented-value"),
    ("percent", "NOT IN RECORD. She received fifty percent.", False, "invented-value"),
    ("percent", "NOT IN RECORD. She likely got sixty percent.", False, "invented-value"),
    ("percent", "NOT IN RECORD. She received half of the estate.", False, "invented-value"),
    # ... while chatty refusals that invent nothing still pass
    ("name", "NOT IN RECORD. Neither Mother nor Father named one.", True, "marker"),
    ("name", "NOT IN RECORD. Neither the Mother nor the Father identifies a guardian.", True, "marker"),
    ("name", "NOT IN RECORD. The Family Court did not appoint one.", True, "marker"),
    ("name", "NOT IN RECORD. The opinion does not mention a Guardian ad Litem.", True, "marker"),
    ("money", "NOT IN RECORD. One cannot tell the dollar amount from these materials.", True, "marker"),
    ("money", "NOT IN RECORD. The opinion mentions no fee award in dollars.", True, "marker"),
    ("percent", "NOT IN RECORD. No one percentage is stated.", True, "marker"),
    ("date", "NOT IN RECORD. The opinion, filed in 2019, does not mention a marriage.", True, "marker"),
    ("date", "NOT IN RECORD. The materials do not say when or whether the parties married.", True, "marker"),
    ("date", "NOT IN RECORD (the opinion is dated in the header only).", True, "marker"),
])
def test_refusal(kind: str, reply: str, passed: bool, reason: str) -> None:
    score = score_refusal(reply, kind)  # type: ignore[arg-type]
    assert (score.passed, score.reason) == (passed, reason)
