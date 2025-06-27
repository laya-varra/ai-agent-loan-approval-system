
import { useLocation, useNavigate } from "react-router-dom";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Bot, MessageSquare } from "lucide-react";

const AiResults = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { result, prompt } = location.state || {};

  if (!result) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-orange-50 to-amber-100 flex items-center justify-center">
        <Card className="max-w-md">
          <CardContent className="text-center py-8">
            <p>No results found. Please go back and try again.</p>
            <Button onClick={() => navigate('/ai-suggestions')} className="mt-4">
              Back to AI Suggestions
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 to-amber-100">
      <div className="container mx-auto px-4 py-8">
        <Button 
          onClick={() => navigate('/ai-suggestions')} 
          variant="ghost" 
          className="mb-6"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to AI Suggestions
        </Button>

        <div className="max-w-4xl mx-auto space-y-6">
          <Card className="shadow-lg border-0">
            <CardHeader className="text-center">
              <div className="w-16 h-16 bg-orange-500 rounded-full flex items-center justify-center mx-auto mb-4">
                <Bot className="w-8 h-8 text-white" />
              </div>
              <CardTitle className="text-2xl font-bold">AI Suggestion Results</CardTitle>
              <CardDescription>Powered by Groq AI</CardDescription>
            </CardHeader>
          </Card>

          <Card className="shadow-md">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <MessageSquare className="w-5 h-5 text-blue-500" />
                Your Question
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                <p className="text-sm text-blue-800 font-medium">
                  {prompt}
                </p>
              </div>
            </CardContent>
          </Card>

          <Card className="shadow-md">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Bot className="w-5 h-5 text-orange-500" />
                AI Response
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="bg-orange-50 p-6 rounded-lg border border-orange-200">
                <div className="prose prose-sm max-w-none">
                  <pre className="whitespace-pre-wrap text-sm text-orange-900 font-sans leading-relaxed">
                    {result.suggestion}
                  </pre>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="shadow-md bg-green-50 border-green-200">
            <CardHeader>
              <CardTitle className="text-lg text-green-800">💡 Pro Tips</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-green-700">
              <ul className="space-y-2 ml-4 list-disc">
                <li>Save important suggestions for future reference</li>
                <li>Ask follow-up questions for more detailed advice</li>
                <li>Use specific details about your situation for better recommendations</li>
                <li>Consider consulting with a financial advisor for personalized guidance</li>
              </ul>
            </CardContent>
          </Card>

          <div className="flex justify-center space-x-4">
            <Button 
              onClick={() => navigate('/ai-suggestions')}
              className="bg-orange-500 hover:bg-orange-600"
            >
              Ask Another Question
            </Button>
            <Button variant="outline" onClick={() => navigate('/')}>
              Back to Home
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AiResults;
