
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";
import { CreditCard, FileText, Bot, Building } from "lucide-react";

const Index = () => {
  const navigate = useNavigate();

  const features = [
    {
      title: "PAN Card Verification",
      description: "Extract and verify PAN card details using OCR technology",
      icon: CreditCard,
      path: "/pan-verification",
      color: "bg-blue-500"
    },
    {
      title: "Aadhaar Card Verification", 
      description: "Extract and verify Aadhaar card details with OCR",
      icon: FileText,
      path: "/aadhaar-verification",
      color: "bg-green-500"
    },
    {
      title: "Loan Application",
      description: "Submit and process loan applications with AI agents",
      icon: Building,
      path: "/loan-application",
      color: "bg-purple-500"
    },
    {
      title: "AI Suggestions",
      description: "Get intelligent suggestions powered by Groq AI",
      icon: Bot,
      path: "/ai-suggestions",
      color: "bg-orange-500"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-12">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-900 mb-4">
            🏦 AGENTIC AI BASED LOAN APPROVAL SYSTEM
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Advanced loan processing system with AI-powered document verification, 
            intelligent decision making, and automated approval workflows.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto">
          {features.map((feature, index) => {
            const IconComponent = feature.icon;
            return (
              <Card 
                key={index} 
                className="hover:shadow-lg transition-all duration-300 transform hover:-translate-y-1 cursor-pointer border-0 shadow-md"
                onClick={() => navigate(feature.path)}
              >
                <CardHeader className="text-center pb-2">
                  <div className={`w-16 h-16 ${feature.color} rounded-full flex items-center justify-center mx-auto mb-4`}>
                    <IconComponent className="w-8 h-8 text-white" />
                  </div>
                  <CardTitle className="text-lg font-semibold">
                    {feature.title}
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-center">
                  <CardDescription className="text-sm text-gray-600 mb-4">
                    {feature.description}
                  </CardDescription>
                  <Button 
                    variant="outline" 
                    className="w-full"
                    onClick={(e) => {
                      e.stopPropagation();
                      navigate(feature.path);
                    }}
                  >
                    Get Started
                  </Button>
                </CardContent>
              </Card>
            );
          })}
        </div>

        <div className="mt-16 text-center">
          <div className="bg-white rounded-lg shadow-md p-8 max-w-4xl mx-auto">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">
              How It Works
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-3">
                  <span className="text-blue-600 font-bold">1</span>
                </div>
                <h3 className="font-semibold mb-2">Upload Documents</h3>
                <p className="text-sm text-gray-600">Upload PAN/Aadhaar cards for OCR extraction</p>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-3">
                  <span className="text-green-600 font-bold">2</span>
                </div>
                <h3 className="font-semibold mb-2">AI Processing</h3>
                <p className="text-sm text-gray-600">Multiple AI agents analyze your application</p>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-3">
                  <span className="text-purple-600 font-bold">3</span>
                </div>
                <h3 className="font-semibold mb-2">Get Results</h3>
                <p className="text-sm text-gray-600">Receive instant decisions and suggestions</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
