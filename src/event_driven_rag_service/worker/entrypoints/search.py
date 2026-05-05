"""Search worker entrypoint — now handled by the generic CPU worker.

Run both chunk and search workers with:
    python -m event_driven_rag_service.worker.entrypoints.cpu
"""
from event_driven_rag_service.worker.entrypoints.cpu import main

if __name__ == "__main__":
    main()
