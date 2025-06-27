
import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, Bot } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const AiSuggestions = () => {
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const { toast } = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) {
      toast({
        title: "Error",
        description: "Please enter a prompt",
        variant: "destructive",
      });
      return;
    }

    setLoading(true);

    try {
      const response = await fetch('http://localhost:5000/groq-suggest', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });
      
      const result = await response.json();
      
      if (response.ok) {
        navigate('/ai-results', { state: { result, prompt } });
      } else {
        toast({
          title: "Error",
          description: "Failed to get AI suggestions",
          variant: "destructive",
        });
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to connect to server",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const samplePrompts = [
    "How can I improve my CIBIL score to get better loan terms?",
    "What factors should I consider before applying for a home loan?",
    "Explain the difference between fixed and floating interest rates",
    "What documents are typically required for a personal loan application?",
    "How does debt-to-income ratio affect loan approval?"
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 to-amber-100">
      <div className="container mx-auto px-4 py-8">
        <Button 
          onClick={() => navigate('/')} 
          variant="ghost" 
          className="mb-6"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Home
        </Button>

        <div className="max-w-4xl mx-auto space-y-6">
          <Card className="shadow-lg border-0">
            <CardHeader className="text-center pb-6">
              <div className="w-16 h-16 bg-orange-500 rounded-full flex items-center justify-center mx-auto mb-4">
                <Bot className="w-8 h-8 text-white" />
              </div>
              <CardTitle className="text-2xl font-bold">AI Suggestions</CardTitle>
              <CardDescription>
                Get intelligent financial advice and suggestions powered by Groq AI
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Your Question or Prompt</label>
                  <Textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Ask me anything about loans, credit scores, financial planning..."
                    className="min-h-[120px]"
                  />
                </div>

                <Button 
                  type="submit" 
                  className="w-full bg-orange-500 hover:bg-orange-600" 
                  disabled={loading || !prompt.trim()}
                >
                  {loading ? "Getting AI Suggestions..." : "Get AI Suggestions"}
                </Button>
              </form>
            </CardContent>
          </Card>

          <Card className="shadow-md">
            <CardHeader>
              <CardTitle className="text-lg">Sample Questions</CardTitle>
              <CardDescription>Click on any question to use it as your prompt</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {samplePrompts.map((samplePrompt, index) => (
                  <Button
                    key={index}
                    variant="outline"
                    className="w-full text-left justify-start h-auto p-3 text-wrap"
                    onClick={() => setPrompt(samplePrompt)}
                  >
                    {samplePrompt}
                  </Button>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="shadow-md bg-blue-50 border-blue-200">
            <CardHeader>
              <CardTitle className="text-lg text-blue-800">How to Use AI Suggestions</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-blue-700">
              <ul className="space-y-2 ml-4 list-disc">
                <li>Ask specific questions about loans, credit, and financial planning</li>
                <li>Request explanations of financial terms and concepts</li>
                <li>Get personalized advice based on your financial situation</li>
                <li>Ask for tips to improve your loan application</li>
                <li>Request comparisons between different loan products</li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default AiSuggestions;
