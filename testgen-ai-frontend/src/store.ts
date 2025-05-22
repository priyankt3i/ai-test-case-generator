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
}));

// Types like LLMProviderDetail are now in src/types.ts
