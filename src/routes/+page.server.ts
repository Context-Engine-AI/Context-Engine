import { GITHUB_TOKEN } from '$env/static/private';

const GITHUB_API = 'https://api.github.com/graphql';

const REPO_QUERY = `
  query {
    repository(owner: "m1rl0k", name: "Context-Engine") {
      stargazerCount
      forkCount
      issues(states: OPEN) { 
        totalCount 
      }
      pullRequests(states: OPEN) { 
        totalCount 
      }
      releases(first: 1) {
        nodes {
          name
          publishedAt
          tagName
        }
      }
      languages(first: 5) {
        edges {
          size
          node { 
            name 
            color
          }
        }
      }
      defaultBranchRef {
        target {
          ... on Commit {
            history(first: 1) {
              nodes {
                committedDate
              }
            }
          }
        }
      }
    }
  }
`;

export async function load() {
  try {
    // Only fetch if token is available
    if (!GITHUB_TOKEN || GITHUB_TOKEN === 'your_github_token_here') {
      return {
        metrics: {
          stars: '2.1k',
          forks: '456', 
          language: 'Python',
          status: 'fallback'
        }
      };
    }

    const response = await fetch(GITHUB_API, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${GITHUB_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query: REPO_QUERY }),
    });

    if (!response.ok) {
      throw new Error(`GitHub API error: ${response.status}`);
    }

    const data = await response.json();
    
    if (data.errors) {
      throw new Error(`GraphQL errors: ${JSON.stringify(data.errors)}`);
    }

    const repo = data.data?.repository;
    
    if (!repo) {
      throw new Error('Repository not found');
    }

    // Calculate total language size for percentages
    const totalSize = repo.languages.edges.reduce((sum: number, edge: any) => sum + edge.size, 0);
    
    // Sort languages by size (descending) to get the primary language
    const sortedLanguages = repo.languages.edges.sort((a: any, b: any) => b.size - a.size);
    const primaryLanguage = sortedLanguages[0]?.node.name || 'Unknown';
    
    const metrics = {
      stars: repo.stargazerCount.toLocaleString(),
      forks: repo.forkCount.toLocaleString(),
      openIssues: repo.issues.totalCount,
      openPRs: repo.pullRequests.totalCount,
      language: primaryLanguage,
      latestRelease: repo.releases.nodes[0] || null,
      languages: sortedLanguages.map((edge: any) => ({
        name: edge.node.name,
        color: edge.node.color,
        percentage: ((edge.size / totalSize) * 100).toFixed(1)
      })),
      lastCommit: repo.defaultBranchRef?.target?.history?.nodes[0]?.committedDate || null,
      status: 'live'
    };

    return { metrics };
    
  } catch (error) {
    console.warn('Failed to fetch GitHub metrics:', error);
    return {
      metrics: {
        stars: '2.1k',
        forks: '456', 
        language: 'Python',
        status: 'fallback',
        error: error instanceof Error ? error.message : 'Unknown error'
      }
    };
  }
}