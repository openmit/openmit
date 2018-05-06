#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>
#include "helloworld.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using helloworld::HelloRequest;
using helloworld::HelloReply;
using helloworld::Greeter;

class GreeterClient {
  public:
    GreeterClient(std::shared_ptr<Channel> channel) : stub_(Greeter::NewStub(channel)) {}

    // Assembles the client's payload. sends it and presents the response back from the server.
    // from the server.
    std::string SayHello(const std::string& user) {
      // Data we are sending to the server 
      HelloRequest request;
      request.set_name(user);

      // Container for the data we expect from the server 
      HelloReply reply;

      // Context for the client. 
      ClientContext context;

      // The actual RPC.
      //Status status = stub_->SayHello(&context, request, &reply);
      Status status = stub_->SayNiHao(&context, request, &reply);

      // Act upon its status.
      if (status.ok()) {
        return reply.message();
      } else {
        std::cout << status.error_code() << ": " << status.error_message() << std::endl;
        return "RPC failed.";
      }
    }

  private:
    std::unique_ptr<Greeter::Stub> stub_;
};

int main(int argc, char** argv) {
  GreeterClient greeter(grpc::CreateChannel(
    "localhost:50051", grpc::InsecureChannelCredentials()));
  for (int i = 1; i < 100; i++) {
    std::string user("world. zhouyong+");
    user += std::to_string(i);
    std::string reply = greeter.SayHello(user);
    std::cout << "Greeter received: " << reply << std::endl;
  }
  
  return 0;
}
