import { create } from 'zustand';
import type { LLMProviderDetail } from './types'; // Use type-only import

// Define the shape of your state
interface AppState {
  // LLM Configuration
  llmProviders: Record<string, LLMProviderDetail>;
  availableModels: string[];
  selectedProvider: string | null;
  selectedModel: string | null;
  apiCredentials: Record<string, string>; // E.g., { 'api_key': 'value' }
  openaiFallbackApiKey: string;

  // Document Processing State
  extractedRequirementsText: string | null;
  
  // Application Identification State
  identifiedApplications: string[];
  isIdentifyingApps: boolean;
  identifyAppsError: string | null;

  // Application Context Selection State
  selectedApplications: string[];
  // Stores extracted text for context files: { appName: ["text1", "text2"], ... }
  applicationContexts: Record<string, string[]>; 

  // Test Case Generation State
  generatedTestCases: Record<string, any>; // AppName -> List<TestCase> or ErrorString
  isGeneratingTestCases: boolean;
  generateTestCasesError: string | null;

  // Single Test Case Refactoring State
  isRefactorDialogOpen: boolean;
  refactoringTestCase: { appName: string; tcId: string; originalData: any } | null; // 'any' for now, ideally TestCase
  refactorInstructions: string;
  isRefactoringSingleTc: boolean;
  refactorSingleTcError: string | null;

  // Bulk Test Case Refactoring State
  isBulkRefactorDialogOpen: boolean;
  bulkRefactoringAppName: string | null;
  bulkRefactorInstructions: string;
  isRefactoringBulkTc: boolean;
  refactorBulkTcError: string | null;

  // AI Review - Get Suggestions State
  // Stores review data per app: { appName: { coverage_summary: "...", newly_suggested_test_cases: [...], ... } }
  aiReviewData: Record<string, any>; 
  isPerformingAiReview: Record<string, boolean>; 
  performAiReviewError: Record<string, string | null>; 

  // AI Review - Apply Changes State
  aiReviewUserDecisions: Record<string, Record<string, string>>;
  isApplyingAiReview: Record<string, boolean>; 
  applyAiReviewError: Record<string, string | null>; 

  // Export to Excel State
  isExportingToExcel: boolean;
  exportToExcelError: string | null;

  // Actions to update state
  setLlmProviders: (providers: Record<string, LLMProviderDetail>) => void;
  setSelectedProvider: (provider: string | null) => void;
  setSelectedModel: (model: string | null) => void;
  setApiCredential: (provider: string, key: string, value: string) => void; 
  setOpenAIFallbackApiKey: (key: string) => void;
  updateAvailableModels: () => void; 
  setExtractedRequirementsText: (text: string | null) => void;
  setIdentifiedApplications: (apps: string[]) => void;
  setIsIdentifyingApps: (loading: boolean) => void;
  setIdentifyAppsError: (error: string | null) => void;
  setSelectedApplications: (apps: string[]) => void;
  addApplicationContext: (appName: string, contextFileText: string) => void;
  clearApplicationContexts: (appName?: string) => void; 
  setGeneratedTestCases: (results: Record<string, any>) => void;
  setIsGeneratingTestCases: (loading: boolean) => void;
  setGenerateTestCasesError: (error: string | null) => void;
  openRefactorDialog: (appName: string, tcId: string, originalData: any) => void;
  closeRefactorDialog: () => void;
  setRefactorInstructions: (instructions: string) => void;
  updateSingleTestCase: (appName: string, tcId: string, updatedTcData: any) => void;
  setIsRefactoringSingleTc: (loading: boolean) => void;
  setRefactorSingleTcError: (error: string | null) => void;
  openBulkRefactorDialog: (appName: string) => void;
  closeBulkRefactorDialog: () => void;
  setBulkRefactorInstructions: (instructions: string) => void;
  updateAllTestCasesForApp: (appName: string, updatedTcList: any[]) => void;
  setIsRefactoringBulkTc: (loading: boolean) => void;
  setRefactorBulkTcError: (error: string | null) => void;
  setAiReviewData: (appName: string, reviewData: any) => void;
  setIsPerformingAiReview: (appName: string, loading: boolean) => void;
  setPerformAiReviewError: (appName: string, error: string | null) => void;
  clearAiReviewStateForApp: (appName: string) => void;
  setAiReviewUserDecision: (appName: string, suggestionId: string, decision: string) => void;
  clearAiReviewUserDecisionsForApp: (appName: string) => void;
  updateTestCasesAfterAiReview: (appName: string, updatedTestCases: any[]) => void;
  setIsApplyingAiReview: (appName: string, loading: boolean) => void;
  setApplyAiReviewError: (appName: string, error: string | null) => void;
  setIsExportingToExcel: (loading: boolean) => void;
  setExportToExcelError: (error: string | null) => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  // Initial state
  llmProviders: {},
  availableModels: [],
  selectedProvider: null,
  selectedModel: null,
  apiCredentials: {},
  openaiFallbackApiKey: '',
  extractedRequirementsText: null,
  identifiedApplications: [],
  isIdentifyingApps: false,
  identifyAppsError: null,
  selectedApplications: [],
  applicationContexts: {},
  generatedTestCases: {},
  isGeneratingTestCases: false,
  generateTestCasesError: null,
  isRefactorDialogOpen: false,
  refactoringTestCase: null,
  refactorInstructions: '',
  isRefactoringSingleTc: false,
  refactorSingleTcError: null,
  isBulkRefactorDialogOpen: false,
  bulkRefactoringAppName: null,
  bulkRefactorInstructions: '',
  isRefactoringBulkTc: false,
  refactorBulkTcError: null,
  aiReviewData: {},
  isPerformingAiReview: {},
  performAiReviewError: {},
  aiReviewUserDecisions: {},
  isApplyingAiReview: {},
  applyAiReviewError: {},
  isExportingToExcel: false,
  exportToExcelError: null,

  // Actions
  setLlmProviders: (providers) => set({ 
    llmProviders: providers, 
    selectedProvider: null, // Reset dependent state
    selectedModel: null, 
    availableModels: [],
    apiCredentials: {}, // Reset credentials as they are provider-specific
  }),
  
  setSelectedProvider: (provider) => {
    set({ 
      selectedProvider: provider, 
      selectedModel: null, // Reset model on provider change
      apiCredentials: {} // Reset credentials as they are provider-specific
    }); 
    get().updateAvailableModels();
  },

  setSelectedModel: (model) => set({ selectedModel: model }),

  setApiCredential: (_provider, key, value) => { // provider param might not be needed if creds are generic
    set((state) => ({
      apiCredentials: {
        ...state.apiCredentials, // Keep other credentials if any
        [key]: value,
      },
    }));
  },

  setOpenAIFallbackApiKey: (key) => set({ openaiFallbackApiKey: key }),

  updateAvailableModels: () => {
    const { llmProviders, selectedProvider } = get();
    if (selectedProvider && llmProviders[selectedProvider]) {
      set({ availableModels: llmProviders[selectedProvider].models || [] });
    } else {
      set({ availableModels: [] });
    }
  },

  setExtractedRequirementsText: (text) => set({ 
    extractedRequirementsText: text,
    identifiedApplications: [], // Reset identified apps if new text is set
    identifyAppsError: null,
  }),

  setIdentifiedApplications: (apps) => set((state) => {
    // When identified apps change, reset selected apps and their contexts
    const newSelectedApps = apps.filter(app => state.selectedApplications.includes(app));
    const newAppContexts = { ...state.applicationContexts };
    for (const appName of Object.keys(newAppContexts)) {
      if (!apps.includes(appName)) {
        delete newAppContexts[appName];
      }
    }
    return { 
      identifiedApplications: apps, 
      identifyAppsError: null, 
      isIdentifyingApps: false,
      selectedApplications: newSelectedApps,
      applicationContexts: newAppContexts,
    };
  }),
  setIsIdentifyingApps: (loading) => set({ isIdentifyingApps: loading }),
  setIdentifyAppsError: (error) => set({ identifyAppsError: error, isIdentifyingApps: false, identifiedApplications: [] }),

  setSelectedApplications: (apps) => set((state) => {
    // Prune contexts if an app is deselected
    const newAppContexts = { ...state.applicationContexts };
    for (const appName of Object.keys(newAppContexts)) {
      if (!apps.includes(appName)) {
        delete newAppContexts[appName];
      }
    }
    return { selectedApplications: apps, applicationContexts: newAppContexts };
  }),
  
  addApplicationContext: (appName, contextFileText) => set((state) => ({
    applicationContexts: {
      ...state.applicationContexts,
      [appName]: [...(state.applicationContexts[appName] || []), contextFileText],
    },
  })),

  clearApplicationContexts: (appName) => set((state) => {
    if (appName) {
      const newAppContexts = { ...state.applicationContexts };
      delete newAppContexts[appName];
      return { applicationContexts: newAppContexts };
    }
    return { applicationContexts: {} }; // Clear all
  }),

  setGeneratedTestCases: (results) => set({ generatedTestCases: results, generateTestCasesError: null, isGeneratingTestCases: false }),
  setIsGeneratingTestCases: (loading) => set({ isGeneratingTestCases: loading }),
  setGenerateTestCasesError: (error) => set({ generateTestCasesError: error, isGeneratingTestCases: false, generatedTestCases: {} }),

  openRefactorDialog: (appName, tcId, originalData) => set({ 
    isRefactorDialogOpen: true, 
    refactoringTestCase: { appName, tcId, originalData },
    refactorInstructions: '', // Clear previous instructions
    refactorSingleTcError: null 
  }),
  closeRefactorDialog: () => set({ 
    isRefactorDialogOpen: false, 
    refactoringTestCase: null, 
    refactorInstructions: '',
    isRefactoringSingleTc: false, // Reset loading state
    refactorSingleTcError: null
  }),
  setRefactorInstructions: (instructions) => set({ refactorInstructions: instructions }),
  
  updateSingleTestCase: (appName, tcId, updatedTcData) => set((state) => {
    const appTestCases = state.generatedTestCases[appName];
    if (Array.isArray(appTestCases)) {
      const updatedAppTestCases = appTestCases.map(tc => 
        (tc['Test Case ID'] === tcId || tc.id === tcId) ? { ...tc, ...updatedTcData, 'Test Case ID': tcId } : tc
      );
      return {
        generatedTestCases: {
          ...state.generatedTestCases,
          [appName]: updatedAppTestCases,
        },
        refactorSingleTcError: null,
        isRefactoringSingleTc: false,
        isRefactorDialogOpen: false, // Close dialog on success
        refactoringTestCase: null,
      };
    }
    return {}; // No change if app or test cases not found
  }),
  setIsRefactoringSingleTc: (loading) => set({ isRefactoringSingleTc: loading }),
  setRefactorSingleTcError: (error) => set({ refactorSingleTcError: error, isRefactoringSingleTc: false }),

  openBulkRefactorDialog: (appName) => set({
    isBulkRefactorDialogOpen: true,
    bulkRefactoringAppName: appName,
    bulkRefactorInstructions: '',
    refactorBulkTcError: null,
  }),
  closeBulkRefactorDialog: () => set({
    isBulkRefactorDialogOpen: false,
    bulkRefactoringAppName: null,
    bulkRefactorInstructions: '',
    isRefactoringBulkTc: false,
    refactorBulkTcError: null,
  }),
  setBulkRefactorInstructions: (instructions) => set({ bulkRefactorInstructions: instructions }),

  updateAllTestCasesForApp: (appName, updatedTcList) => set((state) => ({
    generatedTestCases: {
      ...state.generatedTestCases,
      [appName]: updatedTcList,
    },
    refactorBulkTcError: null,
    isRefactoringBulkTc: false,
    isBulkRefactorDialogOpen: false, // Close dialog on success
    bulkRefactoringAppName: null,
  })),
  setIsRefactoringBulkTc: (loading) => set({ isRefactoringBulkTc: loading }),
  setRefactorBulkTcError: (error) => set({ refactorBulkTcError: error, isRefactoringBulkTc: false }),

  setAiReviewData: (appName, reviewData) => set((state) => ({
    aiReviewData: { ...state.aiReviewData, [appName]: reviewData },
    performAiReviewError: { ...state.performAiReviewError, [appName]: null },
    isPerformingAiReview: { ...state.isPerformingAiReview, [appName]: false },
  })),
  setIsPerformingAiReview: (appName, loading) => set((state) => ({
    isPerformingAiReview: { ...state.isPerformingAiReview, [appName]: loading },
    ...(loading && { performAiReviewError: { ...state.performAiReviewError, [appName]: null } }), // Clear error on new attempt
    ...(loading && { aiReviewData: { ...state.aiReviewData, [appName]: null } }), // Clear old data on new attempt
  })),
  setPerformAiReviewError: (appName, error) => set((state) => ({
    performAiReviewError: { ...state.performAiReviewError, [appName]: error },
    isPerformingAiReview: { ...state.isPerformingAiReview, [appName]: false },
  })),
  clearAiReviewStateForApp: (appName) => set((state) => ({
    aiReviewData: { ...state.aiReviewData, [appName]: null },
    isPerformingAiReview: { ...state.isPerformingAiReview, [appName]: false },
    performAiReviewError: { ...state.performAiReviewError, [appName]: null },
    aiReviewUserDecisions: { ...state.aiReviewUserDecisions, [appName]: {} }, // Clear decisions too
  })),

  setAiReviewUserDecision: (appName, suggestionId, decision) => set((state) => ({
    aiReviewUserDecisions: {
      ...state.aiReviewUserDecisions,
      [appName]: {
        ...(state.aiReviewUserDecisions[appName] || {}),
        [suggestionId]: decision,
      },
    },
  })),
  clearAiReviewUserDecisionsForApp: (appName) => set((state) => ({
    aiReviewUserDecisions: {
      ...state.aiReviewUserDecisions,
      [appName]: {},
    },
  })),
  updateTestCasesAfterAiReview: (appName, updatedTestCases) => set((state) => ({
    generatedTestCases: {
      ...state.generatedTestCases,
      [appName]: updatedTestCases,
    },
    applyAiReviewError: { ...state.applyAiReviewError, [appName]: null },
    isApplyingAiReview: { ...state.isApplyingAiReview, [appName]: false },
    // Optionally clear aiReviewData and decisions for the app after applying
    // aiReviewData: { ...state.aiReviewData, [appName]: null }, 
    // aiReviewUserDecisions: { ...state.aiReviewUserDecisions, [appName]: {} },
  })),
  setIsApplyingAiReview: (appName, loading) => set((state) => ({
    isApplyingAiReview: { ...state.isApplyingAiReview, [appName]: loading },
    ...(loading && { applyAiReviewError: { ...state.applyAiReviewError, [appName]: null } }),
  })),
  setApplyAiReviewError: (appName, error) => set((state) => ({
    applyAiReviewError: { ...state.applyAiReviewError, [appName]: error },
    isApplyingAiReview: { ...state.isApplyingAiReview, [appName]: false },
  })),

  setIsExportingToExcel: (loading) => set({ 
    isExportingToExcel: loading,
    ...(loading && { exportToExcelError: null }) // Clear error on new attempt
  }),
  setExportToExcelError: (error) => set({ exportToExcelError: error, isExportingToExcel: false }),
}));

// Types like LLMProviderDetail are now in src/types.ts
