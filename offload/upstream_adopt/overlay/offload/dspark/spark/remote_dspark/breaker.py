# SPDX-License-Identifier: Apache-2.0
"""Round-count circuit breaker of the remote drafter (CONTRACT 10). torch-free.

closed     every round may use the network; consecutive failed attempts are counted.
open       after ``trip`` consecutive failed attempts: the next ``cooldown`` rounds do not
           use the network. They fail fast exactly like a round without a connection
           (capacity: zero drafts; resilience: the local drafter; shadow: no comparison).
half_open  after the cooldown: the next round that sends a STEP is a probe. Success
           closes the breaker; failure opens it again for another ``cooldown`` rounds.

Outcomes come only from rounds that sent a STEP: a failed attempt is a STEP that failed
(timeout, transport error, refused or frame-error reply) or whose requests that needed
drafts all failed. Rounds without a connection, or with nothing to send, are neutral.
Transport failures also keep the client's wall-clock reconnect backoff (client.py); the
breaker bounds what a worker that answers but keeps failing costs (one RTT, or one
deadline, per engine round otherwise). ``trip`` 0 turns the breaker off.

One instance per rank-0 round; touched only by the thread that runs the rounds.
"""

from __future__ import annotations


class CircuitBreaker:
    def __init__(self, trip: int, cooldown: int):
        if trip < 0 or cooldown < 1:
            raise ValueError(f"trip {trip} must be >= 0 and cooldown {cooldown} >= 1")
        self.trip = trip
        self.cooldown = cooldown
        self.state = "closed"
        self.failures = 0      # consecutive failed attempts
        self.left = 0          # rounds still without network while open
        self.trips = 0

    def allow(self) -> bool:
        """Once per round, before any network use: may this round send a STEP?"""
        if self.state != "open":
            return True
        self.left -= 1
        if self.left <= 0:
            self.state = "half_open"
        return False

    def record(self, ok: bool) -> str | None:
        """Outcome of a round that sent a STEP. Returns "tripped" or "recovered" on a
        transition, else None."""
        if self.trip == 0:
            return None
        if ok:
            was = self.state
            self.state, self.failures = "closed", 0
            return "recovered" if was == "half_open" else None
        self.failures += 1
        if self.state == "half_open" or self.failures >= self.trip:
            self.state, self.left = "open", self.cooldown
            self.trips += 1
            return "tripped"
        return None

    def snapshot(self) -> dict:
        return {"state": self.state, "consecutive_failures": self.failures,
                "cooldown_left": self.left if self.state == "open" else 0,
                "trips": self.trips, "trip": self.trip, "cooldown": self.cooldown}
