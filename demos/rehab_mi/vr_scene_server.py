"""Compatibility entry point for the Brainstim VR scene backend."""

from metabci.brainstim.vr import (
    EventHub,
    PAGE,
    get_ipv4_hints,
    main,
    make_handler,
    udp_loop,
)

__all__ = [
    "EventHub",
    "PAGE",
    "get_ipv4_hints",
    "main",
    "make_handler",
    "udp_loop",
]


if __name__ == "__main__":
    main()
