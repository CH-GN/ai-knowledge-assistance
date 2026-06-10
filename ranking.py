import heapq

def get_top_k(results, scores, k=3):
    heap = []

    for i in range(len(results)):
        # use negative score for max-heap behavior
        heapq.heappush(heap, (-scores[i], results[i]))

    top_k = []
    for _ in range(min(k, len(heap))):
        score, result = heapq.heappop(heap)
        top_k.append((result, -score))

    return top_k