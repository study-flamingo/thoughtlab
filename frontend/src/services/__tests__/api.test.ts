import { describe, it, expect, vi, beforeEach } from 'vitest';
import { graphApi } from '../api';
import axios from 'axios';

vi.mock('axios');
const mockedAxios = axios as any;

describe('graphApi', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockedAxios.create.mockReturnValue({
      get: vi.fn(),
      post: vi.fn(),
      interceptors: {
        request: { use: vi.fn() },
        response: { use: vi.fn() },
      },
    });
  });

  it('getFullGraph calls correct endpoint', async () => {
    const mockGet = vi.fn().mockResolvedValue({ data: { nodes: [], edges: [] } });
    mockedAxios.create.mockReturnValue({
      get: mockGet,
      post: vi.fn(),
      interceptors: {
        request: { use: vi.fn() },
        response: { use: vi.fn() },
      },
    });

    await graphApi.getFullGraph();
    expect(mockGet).toHaveBeenCalledWith('/graph/full');
  });

  it('createObservation calls correct endpoint with data', async () => {
    const mockPost = vi.fn().mockResolvedValue({ data: { id: '123' } });
    mockedAxios.create.mockReturnValue({
      get: vi.fn(),
      post: mockPost,
      interceptors: {
        request: { use: vi.fn() },
        response: { use: vi.fn() },
      },
    });

    const observationData = { text: 'Test', confidence: 0.8 };
    await graphApi.createObservation(observationData);

    expect(mockPost).toHaveBeenCalledWith('/nodes/observations', observationData);
  });
});
