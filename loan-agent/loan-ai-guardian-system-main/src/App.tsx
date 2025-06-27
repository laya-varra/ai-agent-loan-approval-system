
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import PanVerification from "./pages/PanVerification";
import PanResults from "./pages/PanResults";
import AadhaarVerification from "./pages/AadhaarVerification";
import AadhaarResults from "./pages/AadhaarResults";
import LoanApplication from "./pages/LoanApplication";
import LoanResults from "./pages/LoanResults";
import AiSuggestions from "./pages/AiSuggestions";
import AiResults from "./pages/AiResults";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/pan-verification" element={<PanVerification />} />
          <Route path="/pan-results" element={<PanResults />} />
          <Route path="/aadhaar-verification" element={<AadhaarVerification />} />
          <Route path="/aadhaar-results" element={<AadhaarResults />} />
          <Route path="/loan-application" element={<LoanApplication />} />
          <Route path="/loan-results" element={<LoanResults />} />
          <Route path="/ai-suggestions" element={<AiSuggestions />} />
          <Route path="/ai-results" element={<AiResults />} />
          {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
