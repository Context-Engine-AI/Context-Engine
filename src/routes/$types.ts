export interface PageData {
	metrics: {
		stars: string;
		forks: string;
		language: string;
		status: string;
		openIssues?: number;
		openPRs?: number;
		languages?: Array<{
			name: string;
			color: string;
			percentage: string;
		}>;
		latestRelease?: {
			name: string;
			publishedAt: string;
			tagName: string;
		} | null;
		lastCommit?: string | null;
		error?: string;
	};
}