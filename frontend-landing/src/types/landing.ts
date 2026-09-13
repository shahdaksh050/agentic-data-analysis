/**
 * Strict TypeScript Definitions for Landing Page Architecture
 * In accordance with /frontend-dev-guidelines and project DESIGN.md specifications.
 */

export type ThemeMode = 'night' | 'day';

export interface ThemePalette {
  readonly stock: string;
  readonly sheet: string;
  readonly sheetAlt: string;
  readonly ink: string;
  readonly graphite: string;
  readonly pen: string;
  readonly penGlow: string;
  readonly risk: string;
  readonly positive: string;
  readonly accent: string;
  readonly grid: string;
  readonly cardBg: string;
  readonly cardBorder: string;
  readonly fogDensity: number;
  readonly exposure: number;
  readonly gridOpacity: number;
}

export interface NavItem {
  readonly id: string;
  readonly label: string;
  readonly href: string;
  readonly isExternal?: boolean;
}

export interface TrustSignal {
  readonly icon: string;
  readonly text: string;
  readonly highlight?: string;
}

export interface PipelineStageInfo {
  readonly stageNumber: number;
  readonly code: string;
  readonly title: string;
  readonly plainTitle: string;
  readonly description: string;
  readonly badges: readonly string[];
  readonly hudLabel: string;
  readonly formationIndex: number;
}

export interface FeatureCard {
  readonly id: string;
  readonly stageTag: string;
  readonly title: string;
  readonly description: string;
  readonly metric: string;
  readonly metricLabel: string;
  readonly category: 'ingestion' | 'statistics' | 'reasoning' | 'machine-learning' | 'safety' | 'synthesis';
}

export interface PricingPlan {
  readonly id: string;
  readonly name: string;
  readonly badge?: string;
  readonly description: string;
  readonly priceMonthly: number | null; // null for custom/enterprise
  readonly priceAnnual: number | null;
  readonly isPopular?: boolean;
  readonly ctaText: string;
  readonly ctaHref: string;
  readonly features: readonly string[];
}

export interface TestimonialItem {
  readonly id: string;
  readonly quote: string;
  readonly author: string;
  readonly role: string;
  readonly organization: string;
  readonly metricHighlight: string;
}

export interface FaqItem {
  readonly question: string;
  readonly answer: string;
  readonly category: 'safety' | 'execution' | 'privacy' | 'data';
}

export interface ThreeMotionTokens {
  readonly dur: {
    readonly micro: number;
    readonly ui: number;
    readonly stage: number;
    readonly cinematic: number;
  };
  readonly damp: {
    readonly cursor: number;
    readonly journey: number;
    readonly velocity: number;
    readonly parallax: number;
  };
  readonly bounds: {
    readonly maxParallaxDeg: number;
    readonly maxCursorImpulse: number;
  };
}

export interface ThreeStageMetrics {
  fps: number;
  drawCalls: number;
  triangles: number;
  instancesCount: number;
  currentScalar: number;
}
