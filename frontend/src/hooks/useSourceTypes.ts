import { useQuery } from '@tanstack/react-query';
import { graphApi } from '../services/api';

// Default source types that are always available as fallback
const DEFAULT_SOURCE_TYPES = [
  "paper",
  "article",
  "book",
  "website",
  "forum",
  "video",
  "podcast",
  "social media",
  "documentation",
  "report",
  "other",
];

/**
 * Hook to fetch all unique source types from the backend.
 * Falls back to default types if the API call fails.
 * Combines backend data with defaults and removes duplicates.
 */
export function useSourceTypes() {
  return useQuery({
    queryKey: ['source-types'],
    queryFn: async () => {
      try {
        const response = await graphApi.getSourceTypes();
        const backendTypes = response.data;

        // Combine with defaults and remove duplicates while preserving order
        const allTypes = Array.from(new Set([...DEFAULT_SOURCE_TYPES, ...backendTypes]));
        return allTypes;
      } catch (error) {
        console.warn('Failed to fetch source types from backend, using defaults:', error);
        // Return defaults on error
        return DEFAULT_SOURCE_TYPES;
      }
    },
    // Cache for 5 minutes
    staleTime: 5 * 60 * 1000,
    // Don't retry on error since we have a fallback
    retry: false,
  });
}

/**
 * Get the default source types (for use when hook is not available)
 */
export function getDefaultSourceTypes(): string[] {
  return [...DEFAULT_SOURCE_TYPES];
}